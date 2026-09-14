#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无 RAG 基线：不检索任何好图/建议文本，只把【原图】发给生成 LLM，
让它“原生”输出 7 个维度的 缺陷分析 + 整改建议，用于对比 RAG 到底有没有用。

产物结构与 RAG 版（output/0907_sv 等）一致：output/<图片名>/solutions.json
  - mode = "no_rag_baseline"
  - evidence_by_dim = {}（无任何检索证据）
  - solutions[i].text 含 “# 拍照缺陷分析” 与 “# 针对性整改建议” 两段，
    段内 “- 画面比例：…” / “- 画面比例整改：…” 按维排布（与 RAG 同格式）
因此可直接复用同一套评价/出图脚本做 A/B：
  python evaluate_qwen.py  --results-dir <本基线输出目录>   # 或 --provider gemini 等
  python eval_report.py    --verdicts-dir <...>_qwen_eval/raw --solutions-dir <本基线输出目录>

生成模型沿用 config 的 VLM_GENERATOR_MODEL（默认 qwen/qwen3-vl-flash-…，DashScope）。
每个“方案”都是一次独立生成（无任何参考）。默认 --n-solutions 1；
设 >1 可得到多次无 RAG 独立生成，用于看模型自身方差。

用法：
  python no_rag_baseline.py --input /workspace/ai-camera-coach-app/backend/test_data/0715/dataset_link/jingxuan   --output   /workspace/ai-ddge/7dimensions/output/0909_no_rag       # 单张
  python no_rag_baseline.py --input /path/to/folder --max-images 3
  python no_rag_baseline.py --input ... --n-solutions 5 --output output/0909_noRAG_x5
默认按 config.IMAGE_EDIT_ENABLED 调用图编（qwen-image）对每份方案生成 solution_N.png；
图编 prompt 取段遵循 config.IMAGE_EDIT_PROMPT_SECTION。不需要图编时设 IMAGE_EDIT_ENABLED=false。
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

from config import (DIM_CN, DIM_ORDER, GENERATE_MAX_NEW_TOKENS,  # noqa: E402
                    GEN_TEMPERATURE, IMAGE_EDIT_ENABLED, IMAGE_EDIT_MODEL,
                    IMAGE_EDIT_PARALLEL, IMAGE_EDIT_PROMPT_SECTION,
                    VLM_GENERATOR_MODEL)
from prompt import GuidanceGenerator, get_qwen_api_pool          # noqa: E402

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.mpo")


def resolve_inputs(input_arg: str) -> list:
    if os.path.isfile(input_arg):
        return [input_arg]
    if os.path.isdir(input_arg):
        paths = []
        for ext in IMAGE_EXTS:
            paths += sorted(glob.glob(os.path.join(input_arg, "**", ext), recursive=True))
        return paths
    raise FileNotFoundError(input_arg)


def build_baseline_prompt(n: int) -> str:
    """无 RAG 指令：只基于原图，禁止编造“参考图/参考文本”。"""
    dims_txt = "、".join(DIM_CN[d] for d in DIM_ORDER)
    return (
        "你是资深摄影与图片审美诊断专家。下面这张是用户实拍照片【唯一输入】，"
        "没有任何检索到的参考好图或参考文本。\n"
        "请【只看这张原图】独立完成两件事：\n"
        "1. 先诊断这张照片在以下 7 个维度的客观问题（基于画面事实，不要臆造画面里不存在的元素）：\n"
        f"   {dims_txt}；\n"
        "2. 再针对每个维度给出具体、可落地、贴合本图的整改建议（拍摄参数/机位/构图/姿态引导/后期均可，"
        "必须针对本图现状，不要空泛套话，也不要引用任何“参考图/参考文本”）。\n"
        "输出请严格使用以下格式（每维一行，7 维齐全，禁止合并/遗漏/额外字段）：\n"
        "# 拍照缺陷分析\n"
        "- 画面比例：xxx\n- 构图取景：xxx\n- 机位视角：xxx\n- 主体位置：xxx\n"
        "- 姿态动作：xxx\n- 对焦景深：xxx\n- 色彩光影：xxx\n\n"
        "# 针对性整改建议\n"
        "- 画面比例整改：xxx\n- 构图取景整改：xxx\n- 机位视角整改：xxx\n"
        "- 主体位置整改：xxx\n- 姿态动作整改：xxx\n- 对焦景深整改：xxx\n- 色彩光影整改：xxx"
    )


def generate_one(generator: GuidanceGenerator, user_img: Image.Image, text_prompt: str,
                 max_new_tokens: int, temperature: float) -> str:
    messages = [{"role": "user", "content": [
        {"type": "text", "text": text_prompt},
        {"type": "image", "image": user_img},
    ]}]
    if generator.provider in ("qwen", "api"):
        pool = get_qwen_api_pool()
        fut = pool.submit(lambda key: "".join(generator.generate_stream(
            messages, max_new_tokens=max_new_tokens,
            temperature=temperature, api_key=key)))
        return fut.result()
    return "".join(generator.generate_stream(messages, max_new_tokens=max_new_tokens,
                                             temperature=temperature))


def _advice_section(text: str) -> str:
    """截取“# 针对性整改建议”到结尾（图编 prompt 用；找不到则用全文）。"""
    for marker in ("# 针对性整改建议", "# 针对性整改"):
        idx = text.find(marker)
        if idx != -1:
            return text[idx:]
    return text


def process_image(generator, img_path: str, out_dir: str,
                  n_solutions: int, max_new_tokens: int, temperature: float,
                  image_edit_enabled: bool = True) -> str:
    user_img = Image.open(img_path).convert("RGB")
    stem = os.path.splitext(os.path.basename(img_path))[0]
    folder = os.path.join(out_dir, stem)
    os.makedirs(folder, exist_ok=True)

    solutions = []
    errors = []
    t0 = time.time()
    text_prompt = build_baseline_prompt(n_solutions)
    for k in range(1, n_solutions + 1):
        try:
            txt = generate_one(generator, user_img, text_prompt,
                               max_new_tokens=max_new_tokens, temperature=temperature).strip()
            solutions.append({
                "index": k,
                "direction": f"无RAG基线·仅原图 第{k}次",
                "text": f"# 方案{k}（无RAG基线）\n" + txt,
                "reference_images": {},
            })
            print(f"[baseline] {stem} 方案{k}/{n_solutions} 完成（{len(txt)} 字）", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[baseline] {stem} 方案{k} 失败: {e}", flush=True)
            errors.append({"solution": k, "error": str(e)})
            solutions.append({"index": k, "direction": f"无RAG基线·仅原图 第{k}次",
                              "text": "", "error": str(e)})

    # ---- 图编：按每份成功的无 RAG 方案对原图做实际编辑（与 RAG 管线一致，可对比编辑效果）----
    edit_paths = []
    edit_s = 0.0
    mode = "no_rag_baseline"
    if image_edit_enabled:
        from image_edit import edit_image_parallel, save_edit_image
        good = [s for s in solutions if s.get("text")]
        prompts = []
        for s in good:
            body = s["text"]
            if IMAGE_EDIT_PROMPT_SECTION == "advice":
                body = _advice_section(body)
            prompts.append(
                "请对这张照片按以下方案进行后期编辑（保持主体与场景不变，只调整摄影审美因素）：\n"
                f"【方案{s['index']}·{s['direction']}】\n{body}")
        t_e = time.time()
        try:
            edited = edit_image_parallel(user_img, prompts, parallel=IMAGE_EDIT_PARALLEL)
        except Exception as e:  # noqa: BLE001
            print(f"[baseline] 图编失败（本次无编辑图）: {e}", flush=True)
            edited = []
        for i, (res, s) in enumerate(zip(edited, good), start=1):
            ep = os.path.join(folder, f"solution_{i}.png")
            try:
                save_edit_image(res, ep)
            except Exception as e:  # noqa: BLE001
                print(f"[baseline] 图编保存失败 {ep}: {e}", flush=True)
                continue
            s["edit_image"] = ep
            edit_paths.append(ep)
        edit_s = time.time() - t_e
        mode = "no_rag_baseline_plus_image_edit"
        print(f"[baseline] {stem} 图编完成 {len(edit_paths)} 张（耗时 {edit_s:.1f}s）", flush=True)

    payload = {
        "input_image": img_path,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "model": generator.model_spec,
        "vlm_generator_mode": generator.provider,
        "evidence_by_dim": {},
        "solutions": solutions,
        "errors": errors,
        "edit_images": edit_paths,
        "timing_s": {"vlm_generation": round(time.time() - t0, 2),
                     "image_edit": round(edit_s, 2)},
    }
    jp = os.path.join(folder, "solutions.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return jp


def main() -> None:
    ap = argparse.ArgumentParser(description="无 RAG 基线：仅原图 -> 7 维分析+建议")
    ap.add_argument("--input", required=True, help="图片路径或文件夹")
    ap.add_argument("--output", default="output/0909_noRAG", help="结果根目录")
    ap.add_argument("--max-images", type=int, default=0, help="最多处理前 N 张（0=全部）")
    ap.add_argument("--n-solutions", type=int, default=1, help="每张独立生成几份（无RAG重复，看方差）")
    ap.add_argument("--max-new-tokens", type=int, default=GENERATE_MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=GEN_TEMPERATURE)
    ap.add_argument("--model-spec", default=None, help="覆盖生成模型（默认 config.VLM_GENERATOR_MODEL）")
    ap.add_argument("--image-edit", action="store_true", default=None,
                    help="执行图编（缺省时遵循 config.IMAGE_EDIT_ENABLED；无 --no-image-edit 时关闭图编可置 IMAGE_EDIT_ENABLED=false）")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    paths = resolve_inputs(args.input)
    if args.max_images > 0:
        paths = paths[:args.max_images]
    if not paths:
        raise SystemExit(f"[baseline] 没找到图片: {args.input}")

    gen = GuidanceGenerator(model_spec=args.model_spec) if args.model_spec else GuidanceGenerator()
    image_edit_enabled = IMAGE_EDIT_ENABLED if args.image_edit is None else args.image_edit
    print(f"[baseline] 生成模型 provider={gen.provider} model={gen.model_spec} | "
          f"图片 {len(paths)} 张 × 每张 {args.n_solutions} 次 | 图编={image_edit_enabled} | out={args.output}")

    done = 0
    for i, p in enumerate(paths, 1):
        try:
            jp = process_image(gen, p, args.output, args.n_solutions,
                               args.max_new_tokens, args.temperature,
                               image_edit_enabled=image_edit_enabled)
            print(f"[baseline] ({i}/{len(paths)}) {p} -> {jp}", flush=True)
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"[baseline] ({i}/{len(paths)}) 失败 {p}: {e}", flush=True)
    print(f"[baseline] 完成：{done}/{len(paths)} 张 -> {args.output}")


if __name__ == "__main__":
    main()
