#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT 视觉评估：基于【原图】核对每张测试图每份方案、每个维度的
“缺陷分析”(事实类) 与 “整改建议”(规范类) 是否正确。

数据来源：
  results-dir（默认 output/0907_sv）下每个“图片名文件夹/solutions.json”，其中：
    - input_image       ：用户原图（将被编辑的照片，也是判定的唯一视觉依据）
    - solutions[i].text ：方案正文，含 “# 拍照缺陷分析” 与 “# 针对性整改建议” 两段，
                           段内为 “- 画面比例：…” 等按维度排布的条目（每方案 7 维）。

流程：
  1) 对每个方案 text 把 7 个维度的【缺陷分析】与【整改建议】分别提取、按维排列；
  2) 把原图 + 所有方案的结构化文本发给 OpenAI 视觉模型（backend/.env 的 OPENAI_API_KEY），
     让 GPT 按原图逐方案逐维给出 analysis/advice 的 correct|wrong|partial + reason；
  3) 汇总输出，便于：
       - 按维度统计正确率（dimension_stats.csv）
       - 按图片统计正确的维度数 / 全对方案数（image_stats.csv）

模型默认 gpt-4o-mini（可用 --model 覆盖）。每张图 1 次请求（含全部方案），已评估的图会缓存
到 out-dir/raw/<图片名>/verdicts.json，重跑自动跳过（--force 可重评）。

访问 OpenAI 需走代理（本机默认 http://127.0.0.1:30000，可用 --proxy 覆盖）；
OpenAI key 无余额时可用 --provider gemini（key 读 backend/.env 的 GEMINI_API_KEY，可达且当前有额度）。

cd /workspace/ai-ddge/7dimensions/show
HTTPS_PROXY=http://127.0.0.1:30000 python evaluate_gpt.py \
  --input-image /workspace/ai-camera-coach-app/backend/test_data/0715/flower100_20  \
  --results-dir /workspace/ai-ddge/7dimensions/output/0910_sv_FIRST_GOOD_PER_POOR_qwen38flash1 \
  --out-dir /workspace/ai-ddge/7dimensions/output/0910_sv_FIRST_GOOD_PER_POOR_qwen38flash1/gemini2.5flash_eval \
  --provider gemini --model gemini-2.5-flash --concurrency 3

（原图默认从 solutions.json 的 input_image 字段读；找不到时可用 --input_image 指定原图文件
  或它的目录，也可用 --data-root 指定回退根目录。）

cd /workspace/ai-ddge/7dimensions/show
HTTPS_PROXY=http://127.0.0.1:30000 python evaluate_gpt.py \
  --results-dir /workspace/ai-ddge/7dimensions/output/0907_sv \
  --out-dir     /workspace/ai-ddge/7dimensions/output/0907_sv/gpt54_eval \
  --provider openai --model gpt-5.4 --concurrency 3 --limit 10

"""



from __future__ import annotations

import argparse
import base64
import csv
import glob
import io
import json
import os
import re
import sys
import time

import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# 维度定义（与 config 一致）
# ---------------------------------------------------------------------------
DIM_ORDER = ["ratio", "composition", "camera", "position", "pose", "focus", "color"]
DIM_CN = {
    "ratio": "画面比例", "composition": "构图取景", "camera": "机位视角",
    "position": "主体位置", "pose": "姿态动作", "focus": "对焦景深", "color": "色彩光影",
}
LABELS = ("correct", "wrong", "partial")

# 方案正文里 “- 中文名/中文名整改：xxx” 的前缀识别（先剥 markdown 加粗，再按冒号前的标签找维度）
_LABEL_RE = re.compile(r"^([^：:]{1,24})[：:]\s*(.*)$")
_NAME2KEY = {
    "画面比例": "ratio", "构图取景": "composition", "机位视角": "camera",
    "主体位置": "position", "姿态动作": "pose", "姿态": "pose",
    "对焦景深": "focus", "对焦": "focus", "色彩光影": "color", "色彩": "color",
}
# 长名优先，避免“色彩光影”被“色彩”抢走
_NAME_ORDER = ["画面比例", "构图取景", "机位视角", "主体位置", "姿态动作",
               "对焦景深", "色彩光影", "姿态", "对焦", "色彩"]

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "output", "0910_sv_FIRST_GOOD_PER_POOR_qwen38flash1")
DEFAULT_ENV_PATH = "/workspace/ai-camera-coach-app/backend/.env"

# ★ 默认评审模型：直接改这两行（也可用 --model 临时覆盖）
#   openai：gpt-5.5（最强）/ gpt-5.4（均衡，默认）/ gpt-5.4-mini（便宜快）/ gpt-4o-mini（最便宜）
#   gemini：gemini-2.5-flash（REST，稳定）/ gemini-3.8-flash（interactions，长输出易截断）
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# 配置 / key
# ---------------------------------------------------------------------------
def load_env_kv(env_path: str) -> dict:
    """读 .env 成 dict（值去引号；export 前缀容忍）。"""
    kv = {}
    if not os.path.exists(env_path):
        return kv
    with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip().strip('"').strip("'")
    return kv


def get_api_key(env_path: str = DEFAULT_ENV_PATH, explicit: str | None = None,
                var: str = "OPENAI_API_KEY") -> str:
    key = explicit or os.environ.get(var, "")
    if not key:
        key = load_env_kv(env_path).get(var, "")
    return key


# ---------------------------------------------------------------------------
# 方案 text -> 每维 {analysis, advice}
# ---------------------------------------------------------------------------
def split_sections(text: str):
    """拆出“拍照缺陷分析”和“针对性整改建议”两段正文。"""
    m_def = text.find("# 拍照缺陷分析")
    m_adv = text.find("# 针对性整改建议")
    defect = text[m_def + len("# 拍照缺陷分析"): m_adv] if m_def != -1 and m_adv > m_def \
        else (text[m_def + len("# 拍照缺陷分析"):] if m_def != -1 else "")
    advice = text[m_adv + len("# 针对性整改建议"):] if m_adv != -1 else ""
    return defect, advice


def parse_bullets(section: str) -> dict:
    """把段内 “- 标签：正文”（可能跨行）解析成 {dim_key: 正文}。
    容错：标签可带 markdown 加粗（**画面比例**）、可带“整改”后缀、可带括号/英文别名。"""
    out: dict = {}
    if not section:
        return out
    # 按 “行首是 - ” 切块；跨行续行拼到当前条目
    bullets: list[str] = []
    for line in section.splitlines():
        if not line.strip():
            continue
        s = line.strip()
        if s[0] in "-*•·":
            bullets.append(s.lstrip("-*•· ").strip())
        elif bullets:
            bullets[-1] += " " + s
    for body in bullets:
        body = re.sub(r"[*_`]+", "", body).strip()      # 去掉 markdown 加粗/斜体
        m = _LABEL_RE.match(body)
        if not m:
            continue
        label, txt = m.group(1).strip(), m.group(2).strip()
        key = next((_NAME2KEY[n] for n in _NAME_ORDER if n in label), None)
        if not key:
            continue
        txt = re.sub(r"^[（(][^）)]*[）)]\s*", "", txt)   # 去掉“（仅使用本组证据）”
        if txt and (key not in out or len(txt) > len(out[key])):
            out[key] = txt
    return out


def parse_solution_dims(text: str) -> dict:
    """返回 {dim_key: {"analysis": str, "advice": str}}。"""
    defect, advice = split_sections(text)
    a = parse_bullets(defect)
    b = parse_bullets(advice)
    return {d: {"analysis": a.get(d, ""), "advice": b.get(d, "")} for d in DIM_ORDER}


# ---------------------------------------------------------------------------
# 图片编码（压缩，控制 token/成本）
# ---------------------------------------------------------------------------
def encode_image(path: str, max_side: int = 900) -> str | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:  # noqa: BLE001
        print(f"[gpt] 原图读取失败 {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Prompt 组装 + OpenAI 调用
# ---------------------------------------------------------------------------
def build_prompt(img_b64: str, solutions: list[dict]) -> list[dict]:
    """把原图 + 每方案 7 维文本排好，组装 chat messages。"""
    sys_prompt = (
        "你是资深摄影与图片审美诊断专家。任务：对照【原图】，逐方案、逐维度核对 "
        "“缺陷分析”（事实类：描述原图当前客观状态）与“整改建议”（规范类：给摄影师的改进指导）是否正确。\n"
        "判定规则（对 analysis 与 advice 分别判定）：\n"
        "- correct：缺陷分析符合原图客观事实；整改建议结合原图场景与通用摄影美学规范合理、可落地；\n"
        "- wrong：缺陷分析描述的问题在原图中不存在、或与事实相反；整改建议不适用本场景/属模型臆造；\n"
        "- partial：部分正确、部分错误/不准确。\n"
        "注意：建议通常引用若干“参考好图/参考文本”中的处理手法，判定时只看该建议对本图是否成立，不要因为没有逐字复现参考图就判 wrong。\n"
        "只输出 JSON，禁止任何解释、前言、markdown 代码块。"
    )

    lines = ["原图（将被编辑的照片）如下，所有判定都对照这张原图。"]
    lines.append("下面按【方案】给出其 7 个维度的 缺陷分析 与 整改建议（从方案正文中按维度提取）。")
    for s in solutions:
        dims = s.get("dims", {})  # {dim: {analysis, advice}}
        lines.append(f"\n【方案{s.get('index','?')}】（{s.get('direction','')}）")
        for d in DIM_ORDER:
            item = dims.get(d) or {}
            an = (item.get("analysis") or "").strip()
            ad = (item.get("advice") or "").strip()
            lines.append(f"  - {d}（{DIM_CN[d]}）")
            lines.append(f"      缺陷分析: {an if an else '（未提供）'}")
            lines.append(f"      整改建议: {ad if ad else '（未提供）'}")
    lines.append(
        "\n请输出严格 JSON，结构：\n"
        '{"dims":[{"solution":1,"dim":"ratio",'
        '"analysis":{"label":"correct|wrong|partial","reason":"..."},'
        '"advice":{"label":"correct|wrong|partial","reason":"..."}}, ...]}\n'
        "要求：\n"
        "1. 每个方案都输出全部 7 个维度（ratio,composition,camera,position,pose,focus,color），"
        "共 " + str(len(solutions) * 7) + " 条，禁止合并、遗漏、乱序、加多余字段；\n"
        "2. 若某方案某维度“缺陷分析/整改建议”标注为（未提供），对应项 label 输出 null、reason 输出空串；\n"
        "3. reason 用 1~2 句中文，指出依据。"
    )
    user_text = "\n".join(lines)

    content = [{"type": "text", "text": user_text}]
    if img_b64:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content},
    ]


def call_gpt(key: str, model: str, messages: list[dict],
             proxy: str | None = None,
             timeout: int = 180, retries: int = 2) -> dict:
    """OpenAI 兼容 chat completions（json_object），返回解析后的 JSON dict。
    proxy: 如 http://127.0.0.1:30000（本机需代理才能访问 api.openai.com）。"""
    url = "https://api.openai.com/v1/chat/completions"
    proxies = None
    if proxy:
        proxies = {"http": proxy, "https": proxy}
    body = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    # 新模型（gpt-5.x / o1/o3/o4 系列）不再接受 max_tokens，要改用 max_completion_tokens；
    # temperature 也可能被拒（下面遇 400 会自动去掉重试）。
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        body["max_completion_tokens"] = 32000
    else:
        body["max_tokens"] = 16000
    last = None
    for attempt in range(retries + 1):
        retriable = False
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                            "Content-Type": "application/json"},
                              json=body, timeout=timeout, proxies=proxies)
            if r.status_code == 200:
                j = r.json()
                txt = j["choices"][0]["message"]["content"] or ""
                u = j.get("usage") or {}
                if u:   # 便于观察 token 用量/成本
                    print(f"[gpt] {model} tokens: in={u.get('prompt_tokens')} "
                          f"out={u.get('completion_tokens')}", flush=True)
                try:
                    return json.loads(txt)
                except Exception as e:  # noqa: BLE001
                    try:                       # 容忍 ```json 围栏
                        return _parse_json_loose(txt)
                    except Exception:  # noqa: BLE001
                        last = f"JSON 解析失败: {e} | 原始片段: {txt[:200]}"
                        retriable = True   # JSON 解析失败重试一次更稳妥
            else:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                low = r.text.lower()
                # 鉴权/余额不足等业务错误不重试，直接抛出便于定位
                if r.status_code in (401, 403) or ("credit" in low or "quota" in low):
                    raise RuntimeError(last)
                if r.status_code == 400 and "max_tokens" in low and "max_completion_tokens" in low:
                    # 换成新参数名重试（gpt-5.x / o 系列）
                    body.pop("max_tokens", None)
                    body["max_completion_tokens"] = 32000
                    retriable = True
                elif r.status_code == 400 and "temperature" in low and "unsupported" in low:
                    body.pop("temperature", None)
                    retriable = True
                else:
                    # 仅 429 限流 / 5xx 视为瞬时错误
                    retriable = r.status_code in (429, 500, 502, 503, 504)
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            last = str(e)
            retriable = True            # 网络等瞬时异常可重试
        if retriable and attempt < retries:
            wait = 2 ** attempt * 3
            print(f"[gpt] 请求失败(第{attempt + 1}/{retries + 1}次): {last}，{wait}s 后重试")
            time.sleep(wait)
            continue
        raise RuntimeError(last or "GPT 调用失败")


# ---------------------------------------------------------------------------
# Gemini 调用（gemini-3.x 走官方 SDK 的 interactions；其余走 REST generateContent）
# ---------------------------------------------------------------------------
def _parse_json_loose(txt: str) -> dict:
    """容忍 ```json 围栏的 JSON 解析。"""
    t = txt.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _gemini_items(messages: list[dict]) -> list[dict]:
    """OpenAI 风格 messages -> interactions 的 input 列表（text / image 混排）。"""
    items: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            items.append({"type": "text", "text": str(msg.get("content", ""))})
            continue
        for item in msg.get("content", []):
            if item.get("type") == "text":
                items.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image_url":
                url = item["image_url"]["url"]
                if url.startswith("data:image/") and ";base64," in url:
                    mime, b64 = url[len("data:"):].split(";base64,", 1)
                else:
                    mime, b64 = "image/jpeg", url
                items.append({"type": "image", "data": b64, "mime_type": mime})
    return items


def call_gemini_38(key: str, model: str, messages: list[dict],
                   proxy: str | None = None,
                   timeout: int = 300, retries: int = 3,
                   max_output_tokens: int = 32768) -> dict:
    """gemini-3.x（如 gemini-3.8-flash）：走官方 SDK 的 interactions 接口。

    ⚠️ 实测 models.generate_content 对 gemini-3.8-flash 返回 503（high demand），必须用
       client.interactions.create；需要 google-genai 包 + 本机代理。
    ⚠️ 必须显式给 max_output_tokens：默认输出很短，7 维 JSON 会被截断成坏 JSON
       （报 "Expecting ',' delimiter"）。response_format 让模型直接吐 JSON。
    """
    if proxy:
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            os.environ.setdefault(k, proxy)
    try:
        from google import genai
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError("缺少 google-genai：请 pip install -U google-genai") from e

    client = genai.Client(api_key=key)
    items = _gemini_items(messages)
    kwargs = {"model": model, "input": items,
              # max_output_tokens 必须够大；thinking_level 调到 low（3.8-flash 只接受 medium/low/high，
              # 写 minimal 会 400），否则思考 token 占满输出预算会让 JSON 被截断
              "generation_config": {"max_output_tokens": max_output_tokens,
                                    "thinking_level": "low"},
              "response_format": {"type": "text", "mime_type": "application/json"}}
    last = None
    for attempt in range(retries + 1):
        retriable = False
        try:
            resp = client.interactions.create(**kwargs)
            txt = (getattr(resp, "output_text", "") or "").strip()
            if not txt:
                last = "空响应"
                retriable = True
            else:
                try:
                    return _parse_json_loose(txt)
                except Exception as e:  # noqa: BLE001
                    last = f"JSON 解析失败: {e} | 原始: {txt[:200]}"
                    retriable = True
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            last = f"{type(e).__name__}: {msg[:300]}"
            low = msg.lower()
            if "429" in low or "quota" in low or "rate limit" in low:
                # 免费层日额度（GenerateRequestsPerDayPerProjectPerModel-FreeTier = 20 次/项目/模型/天）
                # 与分钟限流要分开：日额度耗尽时重试没意义，直接快速失败并提示。
                if "perday" in low or "per day" in low or "perdayperprojectpermodel" in low:
                    raise RuntimeError(last + "  （Gemini 免费层日额度：每个项目/模型 20 次/天，已用尽；"
                                              "开通结算升为 paid tier 或等每天重置）")
                m = re.search(r"retry in ([\d.]+)\s*s", msg)
                wait = int(float(m.group(1))) + 2 if m else min(2 ** attempt * 5, 60)
                if attempt < retries:
                    print(f"[gemini] 触发限流，{wait}s 后重试 ({attempt + 1}/{retries + 1})")
                    time.sleep(wait)
                    continue
                raise RuntimeError(last + "  （Gemini 限流：可降 --concurrency 或换有额度的 key）")
            if "400" in low and "response_format" in kwargs:
                # 该模型/接口不接受 response_format 时自动去掉再试
                kwargs.pop("response_format", None)
                retriable = True
            elif "400" in low and "thinking_level" in kwargs.get("generation_config", {}):
                kwargs["generation_config"].pop("thinking_level", None)
                retriable = True
            elif any(t in low for t in ("api key not valid", "permission_denied",
                                        "unauthenticated", "invalid api key")):
                raise RuntimeError(last)
            else:
                retriable = True   # 代理/网络抖动（Server disconnected / 503 / timeout）都重试
        if retriable and attempt < retries:
            wait = 2 ** attempt * 3
            print(f"[gemini] 请求失败(第{attempt + 1}/{retries + 1}次): {last}，{wait}s 后重试")
            time.sleep(wait)
            continue
        raise RuntimeError(last or "Gemini 调用失败")


def call_gemini(key: str, model: str, messages: list[dict],
                proxy: str | None = None,
                timeout: int = 180, retries: int = 2) -> dict:
    """按 OpenAI 风格 messages -> Gemini；gemini-3.x 走 interactions，其余走 REST generateContent。"""
    if model.startswith("gemini-3"):
        return call_gemini_38(key, model, messages, proxy=proxy, timeout=timeout,
                              retries=max(retries, 3))
    # 从 messages 提取 system / 文本 / 图片(base64)
    system = ""
    parts: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            system = (system + "\n" + msg["content"]).strip()
            continue
        for item in msg["content"]:
            if item.get("type") == "text":
                parts.append({"text": item["text"]})
            elif item.get("type") == "image_url":
                url = item["image_url"]["url"]
                if url.startswith("data:image/") and ";base64," in url:
                    mime, b64 = url[len("data:"):].split(";base64,", 1)
                else:
                    mime, b64 = "image/jpeg", url
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 32768,
            "responseMimeType": "application/json",
            # ⚠️ 关掉思考（thinkingBudget=0）：思考 token 会占用 maxOutputTokens 预算，
            #    导致 20~35 条判定的 JSON 被截断（报 Unterminated string），且慢十几倍。
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    proxies = {"http": proxy, "https": proxy} if proxy else None
    last = None
    for attempt in range(retries + 1):
        retriable = False
        try:
            r = requests.post(url, params={"key": key}, json=body,
                              timeout=timeout, proxies=proxies)
            if r.status_code == 200:
                j = r.json()
                txt = j["candidates"][0]["content"]["parts"][0]["text"]
                try:
                    return json.loads(txt)
                except Exception as e:  # noqa: BLE001
                    last = f"JSON 解析失败: {e} | 原始: {txt[:200]}"
                    retriable = True
            else:
                err = r.json().get("error", {})
                last = f"HTTP {r.status_code}: {err.get('message', r.text)[:300]}"
                if r.status_code == 400 and "thinkingConfig" in body.get("generationConfig", {}):
                    # 个别模型不支持 thinkingConfig 时自动去掉再试
                    body["generationConfig"].pop("thinkingConfig", None)
                    retriable = True
                    continue
                if r.status_code in (400, 401, 403):
                    raise RuntimeError(last)
                if r.status_code == 429:
                    low = r.text.lower()
                    # 免费层日额度：GenerateRequestsPerDayPerProjectPerModel-FreeTier（20 次/项目/模型/天）
                    if "perday" in low or "per day" in low:
                        raise RuntimeError(last + "  （Gemini 免费层日额度：每个项目/模型 20 次/天，已用尽；"
                                                  "开通结算升为 paid tier 或等每天重置）")
                    retriable = True
                else:
                    retriable = r.status_code in (500, 502, 503, 504)
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            last = str(e)
            retriable = True
        if retriable and attempt < retries:
            wait = 2 ** attempt * 3
            print(f"[gemini] 请求失败(第{attempt + 1}/{retries + 1}次): {last}，{wait}s 后重试")
            time.sleep(wait)
            continue
        raise RuntimeError(last or "Gemini 调用失败")


# ---------------------------------------------------------------------------
# 解析 GPT 输出
# ---------------------------------------------------------------------------
def normalize_verdicts(raw: dict, n_solutions: int) -> list[dict]:
    """把 GPT 输出转成 {solution, dim, analysis:{label,reason}, advice:{...}} 列表。
    只保留 label 合法的；label=null/缺失/非法 -> 该维度判为缺省（不计入统计）。"""
    ok = []
    items = raw.get("dims") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise ValueError("GPT 输出缺少 dims 数组")
    for it in items:
        if not isinstance(it, dict):
            continue
        sol = it.get("solution")
        dim = it.get("dim")
        if dim not in DIM_ORDER or not isinstance(sol, int):
            continue
        an = it.get("analysis") or {}
        ad = it.get("advice") or {}
        def norm(x):
            if not isinstance(x, dict):
                return None
            lab = x.get("label")
            if lab not in LABELS:
                return None
            return {"label": lab, "reason": str(x.get("reason", "")).strip()}
        rec = {"solution": sol, "dim": dim,
               "analysis": norm(an), "advice": norm(ad)}
        if rec["analysis"] or rec["advice"]:
            ok.append(rec)
    if not ok:
        raise ValueError("GPT 输出里没有任何合法维度判定")
    return ok


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def _acc(c, p, w):
    t = c + p + w
    if not t:
        return 0.0, 0.0, 0
    strict = c / t
    weighted = (c + 0.5 * p) / t
    return strict, weighted, t


def aggregate(all_verdicts: list[dict]):
    """all_verdicts: 每图 {image, raw, dims:[{solution,dim,analysis,advice}]}"""
    # dim 统计
    dim_acc = {d: {"analysis": {"correct": 0, "partial": 0, "wrong": 0},
                   "advice": {"correct": 0, "partial": 0, "wrong": 0}}
               for d in DIM_ORDER}
    image_rows = []
    for v in all_verdicts:
        seen_sols: set = set()
        counts = {k: {"analysis": {"correct": 0, "partial": 0, "wrong": 0},
                      "advice": {"correct": 0, "partial": 0, "wrong": 0}}
                  for k in range(1, 99)}
        sol_has_all7 = {}
        for rec in v["dims"]:
            sol = rec["solution"]
            seen_sols.add(sol)
            an, ad = rec["analysis"], rec["advice"]
            if an:
                dim_acc[rec["dim"]]["analysis"][an["label"]] += 1
                counts.setdefault(sol, {}).setdefault("analysis", {}).setdefault(an["label"], 0)
                counts[sol]["analysis"][an["label"]] += 1
            if ad:
                dim_acc[rec["dim"]]["advice"][ad["label"]] += 1
                counts.setdefault(sol, {}).setdefault("advice", {}).setdefault(ad["label"], 0)
                counts[sol]["advice"][ad["label"]] += 1
        # 按图汇总
        ta_c = sum(c["analysis"].get("correct", 0) for c in counts.values())
        ta_p = sum(c["analysis"].get("partial", 0) for c in counts.values())
        ta_w = sum(c["analysis"].get("wrong", 0) for c in counts.values())
        tc_c = sum(c["advice"].get("correct", 0) for c in counts.values())
        tc_p = sum(c["advice"].get("partial", 0) for c in counts.values())
        tc_w = sum(c["advice"].get("wrong", 0) for c in counts.values())
        a_strict, a_weight, a_n = _acc(ta_c, ta_p, ta_w)
        c_strict, c_weight, c_n = _acc(tc_c, tc_p, tc_w)
        # 每个方案 7 维 analysis 全 correct 的个数
        sol_all7 = 0
        sol_all7_any = 0
        for sol, cc in counts.items():
            # 该方案 analysis 判定数==7 且 wrong==0
            n_a = sum(cc["analysis"].get(k, 0) for k in LABELS)
            if n_a == len(DIM_ORDER) and cc["analysis"].get("wrong", 0) == 0 \
                    and cc["analysis"].get("partial", 0) == 0:
                sol_all7 += 1
            if n_a == len(DIM_ORDER) and cc["analysis"].get("wrong", 0) == 0:
                sol_all7_any += 1
        image_rows.append({
            "image": v["image"],
            "solutions_judged": len(seen_sols),
            "analysis_judgments": a_n, "analysis_correct": ta_c,
            "analysis_partial": ta_p, "analysis_wrong": ta_w,
            "analysis_acc_strict": round(a_strict, 4),
            "analysis_acc_weighted": round(a_weight, 4),
            "advice_judgments": c_n, "advice_correct": tc_c,
            "advice_partial": tc_p, "advice_wrong": tc_w,
            "advice_acc_strict": round(c_strict, 4),
            "advice_acc_weighted": round(c_weight, 4),
            "sol_all7_analysis_correct": sol_all7,
            "sol_all7_analysis_no_wrong": sol_all7_any,
        })
    return dim_acc, image_rows


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="GPT 原图维度评估（缺陷分析+整改建议）")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out-dir", default=None,
                    help="评估结果保存目录（默认 <results-dir>_gpt_eval）")
    ap.add_argument("--data-root", action="append", default=[],
                    help="输入原图缺失时的回退根目录（可多次）")
    ap.add_argument("--input-image", "--input_image", dest="input_image", default=None,
                    help="原图文件或其所在目录（可选）：solutions.json 里的 input_image 失效时用它兜底")
    ap.add_argument("--only", default=None, help="只处理含该关键字的图片文件夹")
    ap.add_argument("--limit", type=int, default=0, help="最多处理前 N 张图（0=全部）")
    ap.add_argument("--max-solutions", type=int, default=0, help="每张只取前 N 个方案（0=全部）")
    ap.add_argument("--provider", default="openai", choices=["openai", "gemini"],
                    help="视觉评估后端：openai(gpt-4o-mini 默认) / gemini(gemini-2.5-flash 或 gemini-3.8-flash)")
    ap.add_argument("--model", default=None,
                    help="模型名（默认: openai->gpt-4o-mini, gemini->gemini-2.5-flash；"
                         "openai 推荐 gpt-5.4 / gpt-5.5，或 gpt-5.4-mini 省钱；gemini-3.8-flash 走 interactions）")
    ap.add_argument("--env", default=DEFAULT_ENV_PATH, help="含 OPENAI_API_KEY 的 .env")
    ap.add_argument("--api-key", default=None, help="直接给 key（优先于 .env）")
    ap.add_argument("--proxy", default=None,
                    help="OpenAI 代理，如 http://127.0.0.1:30000（默认依次取 --proxy / "
                         "env OPENAI_PROXY / HTTPS_PROXY / HTTP_PROXY；本机需代理才能访问）")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max-side", type=int, default=900, help="原图压缩边长")
    ap.add_argument("--force", action="store_true", help="忽略缓存强制重评")
    args = ap.parse_args()

    key_var = "GEMINI_API_KEY" if args.provider == "gemini" else "OPENAI_API_KEY"
    default_model = DEFAULT_GEMINI_MODEL if args.provider == "gemini" else DEFAULT_OPENAI_MODEL
    model = args.model or default_model
    key = get_api_key(args.env, args.api_key, var=key_var)
    if not key:
        sys.exit(f"[gpt] 未找到 {key_var}（backend/.env 或 --api-key）")

    proxy = args.proxy or os.environ.get("OPENAI_PROXY") or os.environ.get("HTTPS_PROXY") \
        or os.environ.get("HTTP_PROXY") or "http://127.0.0.1:30000"
    print(f"[gpt] 代理: {proxy} | provider={args.provider} | model={model}")
    roots = list(args.data_root)
    fallback_img = None
    if args.input_image:
        if os.path.isfile(args.input_image):
            fallback_img = args.input_image
        elif os.path.isdir(args.input_image):
            roots.insert(0, args.input_image)   # 目录：当作回退根（按文件名单层匹配）
    out_dir = args.out_dir or (args.results_dir.rstrip("/") + "_gpt_eval")
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    folders = sorted(d for d in glob.glob(os.path.join(args.results_dir, "*", ""))
                     if os.path.isfile(os.path.join(d, "solutions.json")))
    if args.only:
        folders = [d for d in folders if args.only in os.path.basename(os.path.normpath(d))]
    if args.limit > 0:
        folders = folders[:args.limit]
    if not folders:
        sys.exit(f"[gpt] {args.results_dir} 下没有含 solutions.json 的文件夹")

    print(f"[gpt] 待评估图片: {len(folders)} 张 | model={model} | "
          f"max-solutions={args.max_solutions or '全部'} | out={out_dir}")

    todo, done = [], []
    for fd in folders:
        name = os.path.basename(os.path.normpath(fd))
        cache = os.path.join(raw_dir, name, "verdicts.json")
        if os.path.exists(cache) and not args.force:
            try:
                with open(cache, "r", encoding="utf-8") as f:
                    done.append(json.load(f))
                print(f"[gpt] 命中缓存跳过: {name}")
                continue
            except Exception:  # noqa: BLE001
                pass
        todo.append((fd, name))

    def eval_one(fd: str, name: str) -> dict:
        with open(os.path.join(fd, "solutions.json"), "r", encoding="utf-8") as f:
            d = json.load(f)
        img_path = d.get("input_image", "")
        # 回退：--input_image（文件）> --input_image 目录 / --data-root 根目录（按文件名单层/相对路径匹配）
        if img_path and not os.path.exists(img_path):
            base = os.path.basename(img_path)
            cands = ([fallback_img] if fallback_img else []) + [
                os.path.join(root, base) for root in roots
            ] + [os.path.join(root, img_path.lstrip("/")) for root in roots]
            for cand in cands:
                if cand and os.path.exists(cand):
                    img_path = cand
                    break
        sols = [s for s in d.get("solutions", [])
                if isinstance(s, dict) and s.get("text")]
        if args.max_solutions > 0:
            sols = sols[:args.max_solutions]
        # 每个方案提取 7 维文本
        sol_items = []
        for s in sols:
            dims = parse_solution_dims(s.get("text", ""))
            sol_items.append({"index": s.get("index"),
                              "direction": s.get("direction", ""),
                              "dims": dims})
        img_b64 = encode_image(img_path, args.max_side)
        if not img_b64:
            raise RuntimeError(f"原图不可用: {img_path}")
        messages = build_prompt(img_b64, sol_items)
        if args.provider == "gemini":
            raw = call_gemini(key, model, messages, proxy=proxy)
        else:
            raw = call_gpt(key, model, messages, proxy=proxy)
        verdicts = normalize_verdicts(raw, len(sol_items))
        expected = len(sol_items) * len(DIM_ORDER)
        if len(verdicts) < expected:
            print(f"[gpt] 注意: {name} 模型只返回 {len(verdicts)}/{expected} 条判定"
                  f"（缺失的未计入统计）", flush=True)
        rec = {"image": name, "input_image": img_path, "raw": raw, "dims": verdicts}
        os.makedirs(os.path.join(raw_dir, name), exist_ok=True)
        with open(os.path.join(raw_dir, name, "verdicts.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        n_dim = len(verdicts)
        print(f"[gpt] 完成: {name}（{n_dim} 个维度判定）")
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(eval_one, fd, name): name for fd, name in todo}
        for fut in as_completed(futs):
            try:
                done.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"[gpt] 评估失败 {futs[fut]}: {e}")
    print(f"[gpt] 评估耗时 {time.time() - t0:.1f}s，共 {len(done)} 张")

    if not done:
        sys.exit("[gpt] 没有可用的评估结果")

    dim_acc, image_rows = aggregate(done)

    # ---- 维度统计 ----
    dim_rows = []
    print("\n================ 分维度正确率 ================")
    print(f"{'dim':12s} | {'类型':6s} | {'correct':>8s} {'partial':>8s} {'wrong':>8s} "
          f"{'总数':>5s} | {'strict':>7s} {'weighted':>8s}")
    for d in DIM_ORDER:
        for kind in ("analysis", "advice"):
            cc = dim_acc[d][kind]
            strict, weight, tot = _acc(cc["correct"], cc["partial"], cc["wrong"])
            print(f"{d:12s} | {kind:6s} | {cc['correct']:8d} {cc['partial']:8d} "
                  f"{cc['wrong']:8d} {tot:5d} | {strict:7.3f} {weight:8.3f}")
            dim_rows.append({"dim": d, "kind": kind,
                             "correct": cc["correct"], "partial": cc["partial"],
                             "wrong": cc["wrong"], "total": tot,
                             "acc_strict": round(strict, 4),
                             "acc_weighted": round(weight, 4)})

    # ---- 按图片统计 ----
    print("\n================ 按图片统计（缺陷分析 维度正确数） ================")
    print(f"{'image':30s} | {'sol':>3s} {'judge':>5s} {'ok':>4s} {'strict':>7s} "
          f"{'all7方案数':>8s}")
    for r in sorted(image_rows, key=lambda x: -x["analysis_correct"]):
        print(f"{r['image'][:30]:30s} | {r['solutions_judged']:3d} "
              f"{r['analysis_judgments']:5d} {r['analysis_correct']:4d} "
              f"{r['analysis_acc_strict']:7.3f} {r['sol_all7_analysis_correct']:8d}")

    with open(os.path.join(out_dir, "dimension_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"dim_stats": dim_rows, "image_stats": image_rows},
                  f, ensure_ascii=False, indent=2)
    write_csv(os.path.join(out_dir, "dimension_stats.csv"), dim_rows)
    write_csv(os.path.join(out_dir, "image_stats.csv"), image_rows)
    print(f"\n[gpt] 汇总已保存: {out_dir}/dimension_stats.csv|json, image_stats.csv|json")
    print(f"[gpt] 原始判定: {out_dir}/raw/<图片名>/verdicts.json")


if __name__ == "__main__":
    main()
