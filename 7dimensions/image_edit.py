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

from config import (IMAGE_EDIT_INPUT_MAX_PIXELS, IMAGE_EDIT_MAX_WORKERS,  # noqa: E402
                    IMAGE_EDIT_MODEL, IMAGE_EDIT_SIZE, IMAGE_EDIT_TIMEOUT,
                    QWEN_API_KEYS_CONCURRENCY, qwen_api_key)

DASHSCOPE_MM_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


def _img_to_b64(img) -> str:
    # 图编前降采样到 IMAGE_EDIT_INPUT_MAX_PIXELS 内，避免超大图（如 24MP 广角）被 API 拒绝；
    # 上限不宜过小，否则输入细节丢失会让出图发糊（原为硬编码 1024*1024=1MP）。
    try:
        from encoder import _limit_pixels
        img = _limit_pixels(img, IMAGE_EDIT_INPUT_MAX_PIXELS)
    except Exception:  # noqa: BLE001
        pass
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode()


def _resolve_size(img, size) -> str:
    """把 size 解析成 'W*H'。
    size='auto'（默认）-> 返回 ''，即【不传 size】，由模型按输入图比例自适应：
      实测 4:3 图自动输出 2352*1760 ≈ 4.1MP（优于写死 2048*1536=3.15MP、远优于 1024*1024）。
    其他值原样返回（'x' 兼容写成 '*'）。
    """
    if not size:
        return ""
    s = str(size).strip().lower()
    if s == "auto":
        return ""
    return str(size).replace("x", "*")


def _api_keys() -> list:
    """可用 key 列表（去重、跳过空 key），顺序同 config.QWEN_API_KEYS_CONCURRENCY。
    某个 key 欠费/失效时按此顺序换下一个（与文本生成侧的多 key 池思路一致）。"""
    keys = []
    for k, _c in QWEN_API_KEYS_CONCURRENCY:
        if k and k not in keys:
            keys.append(k)
    return keys or [qwen_api_key()]


def _parse_image(out: dict) -> str:
    """从 DashScope 响应里取出图片（data url / http url / 原始 b64）。"""
    try:
        for c in out["output"]["choices"][0]["message"]["content"]:
            if "image" in c:
                return c["image"]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"图编响应解析失败: {out} ({e})")
    raise RuntimeError(f"图编无图片返回: {out}")


def edit_image(img, prompt: str, model: str = IMAGE_EDIT_MODEL,
               size: str = IMAGE_EDIT_SIZE, max_retries: int = 5,
               timeout: int | None = None) -> str:
    """调用图编模型编辑单张图，返回图片数据（data url / http url / 原始 b64）。

    多 key 兜底（按 QWEN_API_KEYS_CONCURRENCY 顺序）：
      · 某 key 欠费（HTTP 400 Arrearage）/ 鉴权失败（401/403）→ 立即换下一个 key；
      · 429/5xx（限流/服务暂不可用）→ 同一 key 指数退避重试；
      · 其它错误 → 记录后换下一个 key。
    所有 key 都失败才抛异常。
    """
    model_id = model.split("/")[-1]
    size_str = _resolve_size(img, size)
    # watermark 属于 parameters（官方示例放在 body 顶层是无效的；默认不加水印）
    params: dict = {"watermark": False}
    if size_str:
        params["size"] = size_str
    body = {
        "model": model_id,
        "input": {"messages": [{"role": "user", "content": [
            {"image": _img_to_b64(img)},
            {"text": prompt},
        ]}]},
        "parameters": params,
    }

    req_timeout = timeout or IMAGE_EDIT_TIMEOUT
    keys = _api_keys()
    last_err = None
    for ki, key in enumerate(keys, start=1):
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    DASHSCOPE_MM_URL,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    json=body, timeout=req_timeout,
                )
            except requests.exceptions.RequestException as e:
                # 网络类异常（含 ReadTimeout）：记为该 key 本次失败，换下一个 key 再试
                last_err = (f"key#{ki} 请求异常 {type(e).__name__}: {str(e)[:120]}"
                            f"（timeout={req_timeout}s，可调大 IMAGE_EDIT_TIMEOUT）")
                print(f"[image_edit] {last_err}", flush=True)
                break
            if resp.status_code == 200:
                return _parse_image(resp.json())

            text = resp.text or ""
            # 欠费 / 鉴权失败：该 key 不可用，直接换下一个（不浪费重试时间）
            if resp.status_code in (401, 403) or (resp.status_code == 400 and "Arrearage" in text):
                last_err = f"key#{ki} 不可用 HTTP {resp.status_code}: {text[:160]}"
                print(f"[image_edit] {last_err}，换下一个 key", flush=True)
                break
            # 限流 / 服务暂不可用：同一 key 退避重试
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = min(2 ** attempt * 2, 30)  # 2s,4s,8s,16s,30s
                print(f"[image_edit] HTTP {resp.status_code}，{wait}s 后重试 "
                      f"({attempt + 1}/{max_retries})", flush=True)
                time.sleep(wait)
                continue
            # 其它错误（如参数问题）：记录后换下一个 key
            last_err = f"key#{ki} HTTP {resp.status_code}: {text[:160]}"
            break
    raise RuntimeError(f"图编失败（已尝试 {len(keys)} 个 key）：{last_err}")


def edit_image_parallel(img, prompts: list, parallel: bool = True,
                        max_workers: int | None = None) -> list:
    """多方案并行图编（并发数默认取 config.IMAGE_EDIT_MAX_WORKERS，避免 429 限流）。

    单个方案失败（超时/欠费/解析失败）只让该项返回 None，**不影响其它方案**——
    避免一个方案超时（ReadTimeout）把整张图已成功的编辑图也一起丢弃。
    返回列表与 prompts 等长，失败项为 None。
    """
    if max_workers is None:
        max_workers = IMAGE_EDIT_MAX_WORKERS
    if not parallel or len(prompts) <= 1:
        out = []
        for p in prompts:
            try:
                out.append(edit_image(img, p))
            except Exception as e:  # noqa: BLE001
                print(f"[image_edit] 单个方案图编失败（跳过该张）: {e}", flush=True)
                out.append(None)
        return out
    with ThreadPoolExecutor(max_workers=min(max_workers, len(prompts))) as ex:
        futs = [ex.submit(edit_image, img, p) for p in prompts]
        out = []
        for f in futs:                      # 按 prompts 顺序取结果，保持对应关系
            try:
                out.append(f.result())
            except Exception as e:  # noqa: BLE001
                print(f"[image_edit] 单个方案图编失败（跳过该张）: {e}", flush=True)
                out.append(None)
        return out


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
