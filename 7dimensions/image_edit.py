# -*- coding: utf-8 -*-
"""
调用远程图编模型（qwen-image 系列，DashScope）对用户照片做实际编辑，可并行多方案。

使用纯 requests 直连 DashScope 原生多模态接口，无需安装 openai/dashscope SDK。
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (IMAGE_EDIT_MAX_WORKERS, IMAGE_EDIT_MODEL,  # noqa: E402
                    IMAGE_EDIT_SIZE, qwen_api_key)

DASHSCOPE_MM_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


def _img_to_b64(img) -> str:
    # 图编前降采样到 1MP 内，避免超大图（如 48MP 广角）被 API 拒绝 / 处理过慢
    try:
        from encoder import _limit_pixels
        img = _limit_pixels(img, 1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode()


def edit_image(img, prompt: str, model: str = IMAGE_EDIT_MODEL,
               size: str = IMAGE_EDIT_SIZE, max_retries: int = 5) -> str:
    """调用图编模型编辑单张图，返回图片数据（data url / http url / 原始 b64）。
    对 429/5xx 做指数退避重试（DashScope qwen-image 并发限流严格，多方案并发时 429 常见）。"""
    model_id = model.split("/")[-1]
    body = {
        "model": model_id,
        "input": {"messages": [{"role": "user", "content": [
            {"image": _img_to_b64(img)},
            {"text": prompt},
        ]}]},
        "parameters": {"size": size.replace("x", "*")} if size else {},
        "watermark": False,
    }
    for attempt in range(max_retries):
        resp = requests.post(
            DASHSCOPE_MM_URL,
            headers={"Authorization": f"Bearer {qwen_api_key()}",
                     "Content-Type": "application/json"},
            json=body, timeout=300,
        )
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
            wait = min(2 ** attempt * 2, 30)  # 退避 2s,4s,8s,16s,30s（qwen-image 限流窗口可能较长）
            print(f"[image_edit] HTTP {resp.status_code}，{wait}s 后重试 ({attempt + 1}/{max_retries})",
                  flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    out = resp.json()
    try:
        for c in out["output"]["choices"][0]["message"]["content"]:
            if "image" in c:
                return c["image"]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"图编响应解析失败: {out} ({e})")
    raise RuntimeError(f"图编无图片返回: {out}")


def edit_image_parallel(img, prompts: list, parallel: bool = True,
                        max_workers: int | None = None) -> list:
    """多方案并行图编（并发数默认取 config.IMAGE_EDIT_MAX_WORKERS，避免 429 限流）。"""
    if max_workers is None:
        max_workers = IMAGE_EDIT_MAX_WORKERS
    if not parallel or len(prompts) <= 1:
        return [edit_image(img, p) for p in prompts]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(prompts))) as ex:
        return list(ex.map(lambda p: edit_image(img, p), prompts))


def save_edit_image(img_data: str, path: str) -> None:
    """把图编结果（data url / http url / 原始 b64）存成文件。"""
    if img_data.startswith("data:"):
        raw = base64.b64decode(img_data.split(",", 1)[1])
    elif img_data.startswith("http"):
        r = requests.get(img_data, timeout=120)
        r.raise_for_status()
        raw = r.content
    else:
        raw = base64.b64decode(img_data)
    with open(path, "wb") as f:
        f.write(raw)
    print(f"[image_edit] 已保存编辑结果: {path}")


if __name__ == "__main__":
    # 自测：python image_edit.py <img_path> <prompt>
    from PIL import Image
    img = Image.open(sys.argv[1]).convert("RGB")
    res = edit_image(img, sys.argv[2] if len(sys.argv) > 2 else "改善构图")
    save_edit_image(res, "edit_test.png")
