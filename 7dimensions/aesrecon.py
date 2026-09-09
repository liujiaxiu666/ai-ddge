# -*- coding: utf-8 -*-
"""
AesRecon 数据解析：content -> 7 个独立维度文本片段。

AesRecon 每条样本 = 参考图片(good) + content(caption)。
content 形如：
  <Ratio>...</Ratio><Composition>...</Composition><Camera>...</Camera>
  <Position>...</Position><Pose>...</Pose><Focus>...</Focus><Color>...</Color>
本模块把 content 拆成 {dim: 该维度原始文本}，供入库元数据 / 生成阶段证据使用。
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AESRECON_METADATA, AESRECON_ROOT, DIM_ORDER, DIM_TAGS  # noqa: E402


def split_content_7dims(content: str) -> dict:
    """把 AesRecon content 拆成 {dim: 该维度原始文本}，缺失维度给空串。"""
    result = {}
    for dim, tag in DIM_TAGS.items():
        m = re.search(rf"<{tag}>(.*?)</{tag}>", content, flags=re.S)
        result[dim] = m.group(1).strip() if m else ""
    return result


def load_aesrecon_samples(metadata_path: str = AESRECON_METADATA,
                          root: str = AESRECON_ROOT,
                          start: int = 0, end: int | None = None,
                          limit: int | None = None) -> list:
    """读取 metadata.jsonl，返回样本列表。

    每条样本：
      {
        "sample_id":     "1",
        "image_path":    参考好图绝对路径（good image）,
        "control_path":  用户侧坏图绝对路径（poor image，调试比对用）,
        "dim_texts":     {ratio/composition/...: 该维度原始文本},
      }

    metadata.jsonl 里的 image_path/control_path 是相对 AesRecon 根目录的
    （如 AesRecon_dataset/images/good_images/1_good.jpg），这里拼成绝对路径。
    """
    samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            img_rel = rec["image_path"]
            ctl_rel = rec.get("control_path", "")
            sample_id = os.path.splitext(os.path.basename(img_rel))[0]

            samples.append({
                "sample_id": sample_id,
                "image_path": os.path.join(root, img_rel) if not os.path.isabs(img_rel) else img_rel,
                "control_path": os.path.join(root, ctl_rel) if ctl_rel and not os.path.isabs(ctl_rel) else ctl_rel,
                "dim_texts": split_content_7dims(rec["caption"]),
            })

    if end is not None:
        samples = samples[start:end]
    else:
        samples = samples[start:]
    if limit is not None:
        samples = samples[:limit]
    return samples


def build_variant_map(metadata_path: str = AESRECON_METADATA,
                      root: str = AESRECON_ROOT) -> dict:
    """坏图 -> 该坏图对应的【全部】good 变体（含各维文本与好图路径）。
    返回 {control_image_path: [{sample_id, good_image_path, dim_texts}, ...]}
    用于“检测到一个坏图后，把它的 good 图该维度的所有指导都加进来”。
    """
    vmap = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ctl = rec["control_path"]
            if not os.path.isabs(ctl):
                ctl = os.path.join(root, ctl)
            good = rec["image_path"]
            if not os.path.isabs(good):
                good = os.path.join(root, good)
            sample_id = os.path.splitext(os.path.basename(rec["image_path"]))[0]
            vmap.setdefault(ctl, []).append({
                "sample_id": sample_id,
                "good_image_path": good,
                "dim_texts": split_content_7dims(rec["caption"]),
            })
    return vmap


def build_basename_index(vmap: dict) -> dict:
    """{control 文件名: [variants, ...]}。数据迁移后索引 meta 里可能是旧根路径，
    与当前 build_variant_map 的 key 字符串不匹配，此时按 basename 回退匹配。"""
    bi: dict = {}
    for k, vs in vmap.items():
        bi.setdefault(os.path.basename(k), []).extend(vs)
    return bi


def lookup_variants(vmap: dict, control_path: str,
                    basename_index: dict | None = None) -> list:
    """按完整路径取变体；索引 meta 里是旧路径（数据迁移前构建）时，回退按 basename 匹配。"""
    if control_path in vmap:
        return vmap[control_path]
    if basename_index is not None:
        return basename_index.get(os.path.basename(control_path), [])
    return []


if __name__ == "__main__":
    samples = load_aesrecon_samples(limit=3)
    for s in samples:
        print("=" * 60)
        print("sample_id:", s["sample_id"])
        print("image_path:", s["image_path"])
        print("control_path:", s["control_path"])
        for dim in DIM_ORDER:
            print(f"  [{dim}] {s['dim_texts'][dim][:80]}")
