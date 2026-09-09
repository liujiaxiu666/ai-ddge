# -*- coding: utf-8 -*-
"""
推理阶段：
  1) 用户图片（支持自定义单张图片 / 图片文件夹）
  2) 同样并行跑 7 次条件提示 -> 得到 7 套查询 patch 集合
  3) 按维度对齐并行做 ANN + MaxSim 检索
  4) 检索结果按维度和参考样图送入 prompt.py 生成拍照指导并保存结果

三种模式（推荐走 main.py 命令；本文件也可直接跑，最终都输出 JSON 汇总 + 计时）：
  # 模式 1：只检索（不生成 prompt / 不图编）-> output/<图片名>/retrieval_summary.json
  python main.py retrieve --input /path/to/user_photo.jpg
  # 模式 2：检索 + 生成若干份 prompt（每份=每维第 k 张候选好图+文本，方案数动态）-> output/<图片名>/solutions.json
  python main.py infer --input /path/to/user_photo.jpg --n-solutions 5
  # 模式 3：检索 + prompt + 图像编辑（追加 --image-edit）
  python main.py infer --input /path/to/user_photo.jpg --n-solutions 5 --image-edit
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from concurrent.futures import ThreadPoolExecutor

from aesrecon import (build_basename_index, build_variant_map,        # noqa: E402
                      lookup_variants)
from config import (AGG_POOL_TOPK, DEVICE, DIM_ORDER, DTYPE,          # noqa: E402
                    ENCODER_MODEL_PATH, EVIDENCE_FIRST_GOOD_PER_POOR, GENERATE_MAX_NEW_TOKENS,
                    MAX_DIM_EVIDENCE, MIN_SOLUTIONS, N_SOLUTIONS, OUTPUT_DIR,
                    RETRIEVAL_PARALLEL, VARIANTS_PER_DIM_POOR)
from encoder import DdgeEncoder                                      # noqa: E402
from faiss_index import FaissIndex                                   # noqa: E402
from prompt import GuidanceGenerator                                 # noqa: E402

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def resolve_input_paths(input_arg: str) -> list:
    """支持传入单张图片路径，或一个图片文件夹路径。"""
    if os.path.isfile(input_arg):
        return [input_arg]
    if os.path.isdir(input_arg):
        paths = []
        for ext in IMAGE_EXTS:
            paths += sorted(glob.glob(os.path.join(input_arg, "**", ext), recursive=True))
        if not paths:
            raise FileNotFoundError(f"文件夹内没有图片: {input_arg}")
        return paths
    raise FileNotFoundError(f"输入路径不存在: {input_arg}")


def retrieve_dim_evidence(query_sets: dict, client: FaissIndex,
                          variant_map: dict,
                          basename_index: dict | None = None,
                          pool_topk: int = AGG_POOL_TOPK,
                          poor_per_dim: int = VARIANTS_PER_DIM_POOR,
                          min_good: int = MIN_SOLUTIONS,
                          max_dim_evidence: int | None = MAX_DIM_EVIDENCE,
                          parallel: bool = RETRIEVAL_PARALLEL,
                          first_good_per_poor: bool = EVIDENCE_FIRST_GOOD_PER_POOR) -> dict:
    """初心版：每个维度【独立】检索它最匹配的坏图，再展开坏图的 good 变体，
    直到每维凑够至少 min_good（默认 MIN_SOLUTIONS=5）张候选好图 + 该维文本。

    各维度最匹配的坏图可以不同（不一定来自同一张 poor 图）。返回
    evidence_by_dim[dim] = [{dim_original_text, good_image_path, sample_id,
                             poor_image_path, score}, ...]
    列表**有序**：每维第 i 项即「第 i 张候选好图」，供方案生成按序号对齐使用
    （方案 k = 每个维度取第 k 张候选好图 + 对应文本）。

    first_good_per_poor=True 时：每张命中的 poor 图【只保留它的第一张好图】作为一条候选，
    即方案 k = 第 k 张【不同】poor 图的第一张好图（方案间按 poor 去重，适合方案间要求 poor 图不重复）。
    """
    dims = list(query_sets.keys())
    if parallel and len(dims) > 1:
        with ThreadPoolExecutor(max_workers=len(dims)) as ex:
            cands_map = dict(zip(dims, ex.map(
                lambda d: client.search_maxsim(query_sets[d], d, topk=pool_topk), dims)))
    else:
        cands_map = {d: client.search_maxsim(query_sets[d], d, topk=pool_topk) for d in dims}

    evidence_by_dim = {}
    for dim, cands in cands_map.items():
        entries = []
        seen_poor = set()
        for c in cands:
            poor = c.get("control_image_path")
            if not poor or poor in seen_poor:
                continue
            seen_poor.add(poor)
            score = c.get("score", 0.0)      # 该坏图在该维的 MaxSim 相似度（写入 json）
            variants = lookup_variants(variant_map, poor, basename_index)
            if first_good_per_poor:          # 每个 poor 只取“第一张好图” -> 方案天然按 poor 图去重
                variants = variants[:1]
            for v in variants:
                t = (v.get("dim_texts") or {}).get(dim, "")
                if t:
                    entries.append({
                        "dim_original_text": t,
                        "good_image_path": v.get("good_image_path", ""),
                        "sample_id": v.get("sample_id", ""),
                        "poor_image_path": poor,
                        "score": round(score, 4),
                    })
            # 每维凑够至少 min_good 张候选好图即停；poor_per_dim 为最多展开坏图数的兜底
            if len(entries) >= min_good or len(seen_poor) >= poor_per_dim:
                break
        if max_dim_evidence is not None:
            entries = entries[:max_dim_evidence]
        evidence_by_dim[dim] = entries
        if entries:
            top_score = entries[0]["score"]
            tag = " [每poor仅首张好图]" if first_good_per_poor else ""
            print(f"[user] {dim:12s}{tag} 命中坏图 {len(seen_poor)} 张，展开 {len(entries)} 张候选好图"
                  f"（目标 ≥{min_good}），Top MaxSim={top_score}")
    return evidence_by_dim


def _save_summary_json(out_dir: str, payload: dict, filename: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def run_retrieval_only(user_img_path: str,
                      output_dir: str = OUTPUT_DIR):
    """只做检索，不生成 prompt，也不做图像编辑；输出 retrieval_summary.json。"""
    return run_inference(user_img_path, output_dir=output_dir,
                         generate_prompt=False, image_edit=None)


def run_inference(user_img_path: str,
                  output_dir: str = OUTPUT_DIR,
                  max_new_tokens: int = GENERATE_MAX_NEW_TOKENS,
                  n_solutions: int = N_SOLUTIONS,
                  generate_prompt: bool = True,
                  image_edit: bool | None = None):
    """单张用户图片 -> 7 套查询 patch -> 每维独立检索+坏图变体展开 -> 可选生成 prompt / 可选图像编辑。
    generate_prompt=False 时为「只检索」模式（输出 retrieval_summary.json）；
    image_edit=None 时按 config.IMAGE_EDIT_ENABLED 决定是否图编，True 则强制图编。

    返回: (evidence_by_dim, solutions, json_path, edit_paths, retrieval_s)
    """
    user_img = Image.open(user_img_path).convert("RGB")

    encoder = DdgeEncoder(model_path=ENCODER_MODEL_PATH, dtype=DTYPE, device=DEVICE)
    client = FaissIndex()              # 每维一个 faiss index，OpenMP 并行

    variant_map = build_variant_map()
    basename_index = build_basename_index(variant_map)

    t_enc0 = time.time()
    query_sets = encoder.get_dim_patch_sets_parallel(user_img)
    encode_s = time.time() - t_enc0

    t_faiss0 = time.time()
    evidence_by_dim = retrieve_dim_evidence(query_sets, client, variant_map,
                                            basename_index=basename_index)
    faiss_search_s = time.time() - t_faiss0

    retrieval_s = encode_s + faiss_search_s
    print(f"[user] 编码 {encode_s:.2f}s + FAISS 检索 {faiss_search_s:.2f}s = 检索合计 {retrieval_s:.2f}s")

    if not generate_prompt:
        stem = os.path.splitext(os.path.basename(user_img_path))[0]
        out_dir = os.path.join(output_dir, stem)
        payload = {
            "input_image": user_img_path,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "retrieve_only",
            "model": ENCODER_MODEL_PATH,
            "vlm_generator_mode": "n/a",
            "evidence_by_dim": evidence_by_dim,
            "solutions": [],
            "edit_images": [],
            "timing_s": {
                "retrieval": round(retrieval_s, 2),
                "encode": round(encode_s, 2),
                "faiss_search": round(faiss_search_s, 2),
                "vlm_generation": 0.0,
                "image_edit": 0.0,
                "total": round(retrieval_s, 2),
            },
        }
        js_path = _save_summary_json(out_dir, payload, "retrieval_summary.json")
        print(f"[user] 检索结果已保存: {js_path} | 计时(s): 编码 {payload['timing_s']['encode']} | FAISS {payload['timing_s']['faiss_search']} | 检索合计 {payload['timing_s']['retrieval']}")
        return evidence_by_dim, [], js_path, [], retrieval_s

    # 检索/编码已完成：释放本函数对编码模型的局部引用。
    # 模型仍保留在 get_shared_vlm 的全局缓存中，文件夹批处理下一张图编码时
    # 直接命中缓存复用，不会重新加载（编码器 4B 与生成器 2B 可同时常驻）。
    del encoder
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    generator = GuidanceGenerator()
    t_vlm0 = time.time()
    try:
        solutions, js_path, edit_paths = generator.generate_solutions_and_save(
            user_img_path, evidence_by_dim,
            output_dir=output_dir, n_solutions=n_solutions, max_new_tokens=max_new_tokens,
            retrieval_s=retrieval_s, encode_s=encode_s, faiss_search_s=faiss_search_s,
            image_edit_enabled=image_edit,
        )
    except Exception as exc:  # noqa: BLE001
        # 生成失败也要落盘一份 JSON（含各阶段计时），方便查看时延，而不是只打一行错误后丢弃
        vlm_s = time.time() - t_vlm0
        stem = os.path.splitext(os.path.basename(user_img_path))[0]
        out_dir = os.path.join(output_dir, stem)
        payload = {
            "input_image": user_img_path,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "failed",
            "error": str(exc),
            "mode": "retrieve_plus_prompt_failed",
            "model": getattr(generator, "model_spec", ""),
            "vlm_generator_mode": getattr(generator, "provider", ""),
            "evidence_by_dim": evidence_by_dim,
            "solutions": [],
            "edit_images": [],
            "timing_s": {
                "retrieval": round(retrieval_s, 2),
                "encode": round(encode_s, 2),
                "faiss_search": round(faiss_search_s, 2),
                "vlm_generation": round(vlm_s, 2),
                "image_edit": 0.0,
                "total": round(retrieval_s + vlm_s, 2),
            },
        }
        js_path = _save_summary_json(out_dir, payload, "solutions_failed.json")
        print(f"[user] VLM 生成失败，已保存失败 JSON（含计时）: {js_path}\n"
              f"  [user] 计时(s): 检索 {payload['timing_s']['retrieval']} | "
              f"VLM生成(失败前耗时) {payload['timing_s']['vlm_generation']} | "
              f"总计 {payload['timing_s']['total']} | 错误: {exc}", flush=True)
        raise
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return evidence_by_dim, solutions, js_path, edit_paths, retrieval_s


def main() -> None:
    parser = argparse.ArgumentParser(description="用户图片 7 维度 DDGE 推理")
    parser.add_argument("--input", required=True, help="用户图片路径 或 图片文件夹路径")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="结果保存目录")
    parser.add_argument("--max-new-tokens", type=int, default=GENERATE_MAX_NEW_TOKENS)
    parser.add_argument("--n-solutions", type=int, default=N_SOLUTIONS,
                        help="方案数上限（默认不限；实际方案数=各维候选好图数的最小值）")
    parser.add_argument("--max-images", type=int, default=8, nargs="?", const=0,
                        help="文件夹模式最多处理多少张（0 或不给数值 = 不限）")
    parser.add_argument("--retrieve-only", action="store_true", default=False,
                        help="只检索，不生成 prompt/图编（输出 retrieval_summary.json）")
    parser.add_argument("--image-edit", action="store_true", default=None,
                        help="执行图像编辑（缺省时遵循 config.IMAGE_EDIT_ENABLED）")
    args = parser.parse_args()

    paths = resolve_input_paths(args.input)
    if len(paths) > args.max_images:
        paths = paths[:args.max_images]
    print(f"[user] 待推理图片: {len(paths)} 张")
    for i, p in enumerate(paths):
        print(f"\n[user] ({i + 1}/{len(paths)}) {p}")
        _evidence, _solutions, js_path, edit_paths, _retr = run_inference(
            p, output_dir=args.output, max_new_tokens=args.max_new_tokens,
            n_solutions=args.n_solutions,
            generate_prompt=not args.retrieve_only, image_edit=args.image_edit,
        )
        print(f"[user] 已保存: {js_path}（{len(edit_paths)} 张编辑图）")


if __name__ == "__main__":
    main()

