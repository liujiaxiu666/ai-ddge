# -*- coding: utf-8 -*-
"""
离线阶段：AesRecon 样本 -> 7 维度 patch 集合 -> 直接产出 FAISS 索引。

按照注释：
  - 一张样本图循环 7 次编码器（本地 qwen3-vl，encoder-only），
    每次输入「图片 + 对应维度的条件提示」，输出 7 套独立 patch 多向量集合；
  - 默认用「坏图」(control_path) 建索引做检索，与用户照片匹配同类构图问题；
    「好图」(image_path) 作为 prompt 参考样图。可用 config.USE_POOR_AS_INDEX=False 改回好图建索引。
  - 每套 patch 集合绑定维度元数据
    （dim / sample_id / orig_image_path 好图 / control_image_path 坏图 / 该维度原始文本）
    攒到内存后一次性写入 FAISS 索引（每维一个 IndexFlatIP，见 faiss_index.py）。

用法：
  python get_7dimensions.py --limit 100            # 只处理前 100 条
  python get_7dimensions.py --start 0 --end 2000   # 处理 [0, 2000)
  python get_7dimensions.py --faiss-part N         # 并行分段号，最后 faiss-merge 合并
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aesrecon import load_aesrecon_samples                    # noqa: E402
from config import (AESRECON_METADATA, AESRECON_ROOT,         # noqa: E402
                    DEDUP_INDEX_IMAGE, DTYPE, ENCODE_METHOD_LABEL, ENCODE_SHARED_VISION,
                    ENCODER_MODEL, ENCODER_MODEL_PATH, FAISS_INDEX_DIR,
                    IMAGE_MAX_PIXELS, PATCH_POOL_K, USE_POOR_AS_INDEX)
from encoder import DdgeEncoder                               # noqa: E402

FLUSH_EVERY = 10000  # 攒满多少行批量写一次（dim=4096 时约 163MB，需 < gRPC 256MB 上限）


def offline_build(limit: int | None = None,
                  start: int = 0, end: int | None = None,
                  rebuild: bool = False, batch_flush: int = FLUSH_EVERY,
                  faiss_part: int | None = None) -> dict:
    """离线入库主流程：编码 7 维 patch 集合 -> 直接写 FAISS 索引（默认后端）。"""
    samples = load_aesrecon_samples(AESRECON_METADATA, AESRECON_ROOT,
                                    start=start, end=end, limit=limit)
    print(f"[offline] 待处理样本数: {len(samples)}")

    encoder = DdgeEncoder(model_path=ENCODER_MODEL_PATH, dtype=DTYPE)
    print(f"[offline] patch 向量维度: {encoder.hidden_dim}")
    faiss_buffers: dict = {}
    part_tag = "" if faiss_part is None else str(faiss_part)
    n_inserted = 0
    n_empty_dim = 0
    n_skipped = 0
    seen_index = set()          # 去重：一张索引用图只建一份索引
    t_start = time.time()
    for idx, sample in enumerate(samples):
        # 索引用图：默认「坏图」(control_path)，匹配用户照片的同类构图问题；
        # prompt 参考样图始终用「好图」(orig_image_path)
        index_img_path = sample["control_path"] if USE_POOR_AS_INDEX else sample["image_path"]
        if not index_img_path or not os.path.exists(index_img_path):
            n_skipped += 1
            print(f"[offline] 跳过（索引用图缺失）: {index_img_path}")
            continue
        if DEDUP_INDEX_IMAGE and index_img_path in seen_index:
            n_skipped += 1      # 同一坏图已被其他样本建过索引，跳过避免重复
            continue
        seen_index.add(index_img_path)
        img = Image.open(index_img_path).convert("RGB")
        # 一张图并行跑 7 次条件提示 -> 7 套独立 patch 多向量集合（单次 batch 前向）
        patch_sets = encoder.get_dim_patch_sets_parallel(img)

        for dim, patch_set in patch_sets.items():
            if patch_set.shape[0] == 0:
                n_empty_dim += 1
                continue
            meta = {
                "dim": dim,
                "sample_id": sample["sample_id"],
                "orig_image_path": sample["image_path"],       # 好图：prompt 参考样图
                "control_image_path": sample["control_path"],  # 坏图：索引用图
                "dim_original_text": sample["dim_texts"].get(dim, ""),
            }
            vec = patch_set.float().cpu().numpy()              # [N, D] fp32
            fb = faiss_buffers.setdefault(dim, {"vecs": [], "meta": []})
            fb["vecs"].append(vec)
            fb["meta"].append([(meta["sample_id"], meta["orig_image_path"],
                                meta["control_image_path"], meta["dim_original_text"])] * vec.shape[0])

        if (idx + 1) % 20 == 0:
            elapsed = max(time.time() - t_start, 1e-6)
            rate = (idx + 1) / elapsed
            remain = (len(samples) - (idx + 1)) / max(rate, 1e-6)
            n_patch = sum(sum(v.shape[0] for v in fb["vecs"])
                          for fb in faiss_buffers.values())
            print(f"[offline] 进度 {idx + 1}/{len(samples)}，已攒 {n_patch} 行 patch，"
                  f"速率 {rate:.2f} 样本/s，预计剩余 {remain / 60:.1f} min")

    # 把内存里的 patch 集合写成本维 FAISS 索引（每维一个 IndexFlatIP）
    from faiss_index import FaissIndex
    FaissIndex(FAISS_INDEX_DIR).build_from_buffers(faiss_buffers, part_tag=part_tag)

    n_patch = sum(sum(v.shape[0] for v in fb["vecs"])
                  for fb in faiss_buffers.values())
    stats = {
        "samples": len(samples),
        "skipped": n_skipped,
        "patch_rows_inserted": n_patch,
        "empty_dim_sets": n_empty_dim,
        "faiss_part": faiss_part,
    }
    # 溯源 meta：记录该库是 A 还是 B 及建库参数，防止 A/B 索引混淆
    import json
    meta = {
        "method": ENCODE_METHOD_LABEL,
        "encode_shared_vision": bool(ENCODE_SHARED_VISION),
        "encoder_model": ENCODER_MODEL,
        "encoder_model_path": ENCODER_MODEL_PATH,
        "image_max_pixels": IMAGE_MAX_PIXELS,
        "patch_pool_k": PATCH_POOL_K,
        "use_poor_as_index": bool(USE_POOR_AS_INDEX),
        "dedup_index_image": bool(DEDUP_INDEX_IMAGE),
        **{k: stats[k] for k in ("samples", "skipped", "patch_rows_inserted",
                                 "empty_dim_sets", "faiss_part")},
    }
    try:
        with open(os.path.join(FAISS_INDEX_DIR, "index_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[offline] 索引溯源 -> {FAISS_INDEX_DIR}/index_meta.json")
    except Exception as e:  # noqa: BLE001
        print(f"[offline] 写索引溯源 meta 失败({e})")

    print(f"[offline] 完成: {stats}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="AesRecon 7 维度 DDGE 离线入库")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条样本")
    parser.add_argument("--start", type=int, default=0, help="起始样本下标")
    parser.add_argument("--end", type=int, default=None, help="结束样本下标(不含)")
    parser.add_argument("--rebuild", action="store_true", help="先清空旧集合再重建")
    parser.add_argument("--batch-flush", type=int, default=FLUSH_EVERY, help="攒多少行批量写一次")
    parser.add_argument("--faiss-part", type=int, default=None, help="并行重建分段号(0..N-1)")
    args = parser.parse_args()

    offline_build(limit=args.limit, start=args.start, end=args.end,
                  rebuild=args.rebuild, batch_flush=args.batch_flush,
                  faiss_part=args.faiss_part)


if __name__ == "__main__":
    main()
