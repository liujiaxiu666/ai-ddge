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

模型默认 qwen3.8-max（DashScope，--provider qwen）。每张图 1 次请求（含全部方案），
已评估的图会缓存到 out-dir/raw/<图片名>/verdicts.json，重跑自动跳过（--force 可重评）。

DashScope 原生接口直连（无需 SDK/代理），key 读 backend/.env 的 QWEN_API_KEY；
也兼容 openai/gemini（需要本机代理 http://127.0.0.1:30000，且 key 需有余额）。

用法示例：
  python evaluate_qwen.py --limit 10            # DashScope qwen3.8-max，先评前 10 个文件夹
  python evaluate_qwen.py                        # 全量评 output/0907_sv
  python evaluate_qwen.py --results-dir output/0907_sv_FIRST_GOOD_PER_POOR
  python evaluate_qwen.py --max-solutions 2      # 每张只取前 2 个方案
  python evaluate_qwen.py --provider gemini --model gemini-2.5-flash
  python evaluate_qwen.py --provider openai --model gpt-4o
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

# 方案正文里 “- 中文名/中文名整改：xxx” 的前缀识别
_DIM_RE = (r"^(画面比例|构图取景|机位视角|主体位置|姿态动作|姿态|对焦景深|对焦|色彩光影|色彩)"
           r"\s*(?:整改)?\s*[：:]\s*(.*)$")
_NAME2KEY = {
    "画面比例": "ratio", "构图取景": "composition", "机位视角": "camera",
    "主体位置": "position", "姿态动作": "pose", "姿态": "pose",
    "对焦景深": "focus", "对焦": "focus", "色彩光影": "color", "色彩": "color",
}

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "output", "0907_sv")
DEFAULT_ENV_PATH = "/workspace/ai-camera-coach-app/backend/.env"

# DashScope 原生多模态接口（纯 requests，无需 dashscope SDK，也无需代理）
DASHSCOPE_MM_URL = ("https://dashscope.aliyuncs.com/api/v1/services/"
                    "aigc/multimodal-generation/generation")


def _extract_json(text: str) -> dict:
    """从模型返回文本里稳健地取出 JSON（容忍代码块/前后缀）。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        pass
    # 回退：找第一对 { } 平衡区间
    start = t.find("{")
    if start == -1:
        raise ValueError(f"响应中没有 JSON: {text[:200]}")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError(f"JSON 括号不平衡: {text[:200]}")


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
    """把段内 “- 标签：正文”（可能跨行）解析成 {dim_key: 正文}。"""
    out: dict = {}
    if not section:
        return out
    # 按 “行首是 - ” 切块；跨行续行拼到当前条目
    bullets: list[str] = []
    for line in section.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("-"):
            bullets.append(line.lstrip()[1:].strip())
        elif bullets:
            bullets[-1] += " " + line.strip()
    for body in bullets:
        m = re.match(_DIM_RE, body)
        if not m:
            continue
        key = _NAME2KEY.get(m.group(1))
        if not key:
            continue
        txt = m.group(2).strip()
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
        "max_tokens": 16000,
    }
    last = None
    for attempt in range(retries + 1):
        retriable = False
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                            "Content-Type": "application/json"},
                              json=body, timeout=timeout, proxies=proxies)
            if r.status_code == 200:
                txt = r.json()["choices"][0]["message"]["content"]
                try:
                    return json.loads(txt)
                except Exception as e:  # noqa: BLE001
                    last = f"JSON 解析失败: {e} | 原始片段: {txt[:200]}"
                    retriable = True    # JSON 解析失败重试一次更稳妥
            else:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                low = r.text.lower()
                # 鉴权/余额不足等业务错误不重试，直接抛出便于定位
                if r.status_code in (401, 403) or ("credit" in low or "quota" in low):
                    raise RuntimeError(last)
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
# Gemini 调用（把 OpenAI 风格 messages 转成 Gemini generateContent）
# ---------------------------------------------------------------------------
def call_gemini(key: str, model: str, messages: list[dict],
                proxy: str | None = None,
                timeout: int = 180, retries: int = 2) -> dict:
    """按 OpenAI 风格 messages -> Gemini REST 调用，返回解析后的 JSON dict。"""
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
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
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
                if r.status_code in (400, 401, 403):
                    raise RuntimeError(last)
                retriable = r.status_code in (429, 500, 502, 503, 504)
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
# DashScope qwen 调用（原生多模态，纯 requests；把 openai 风格 messages 转成 qwen content）
# ---------------------------------------------------------------------------
def call_qwen(key: str, model: str, messages: list[dict],
              timeout: int = 300, retries: int = 2) -> dict:
    """按 OpenAI 风格 messages -> DashScope MultiModalConversation，返回解析后的 JSON dict。"""
    # 提取 system / 文本 / 图片(data url)
    sys_text = ""
    user_texts: list[str] = []
    image_url: str | None = None
    for msg in messages:
        if msg.get("role") == "system":
            sys_text = (sys_text + "\n" + str(msg.get("content", ""))).strip()
            continue
        for item in msg.get("content", []):
            if item.get("type") == "text":
                user_texts.append(item.get("text", ""))
            elif item.get("type") == "image_url" and not image_url:
                image_url = item["image_url"]["url"]
    if not image_url:
        raise RuntimeError("消息里没有图片")
    full_text = (sys_text + "\n\n" + "\n".join(user_texts)).strip()
    body = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [
            {"image": image_url},
            {"text": full_text},
        ]}]},
        "parameters": {"result_format": "message"},
    }
    last = None
    for attempt in range(retries + 1):
        retriable = False
        try:
            r = requests.post(DASHSCOPE_MM_URL,
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"},
                              json=body, timeout=timeout)
            if r.status_code == 200:
                j = r.json()
                content = j["output"]["choices"][0]["message"]["content"]
                text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
                return _extract_json(text)
            else:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                # 鉴权/余额/模型名等业务错误不重试
                low = r.text.lower()
                if r.status_code in (400, 401, 403) or ("invalid" in low and "model" in low) \
                        or ("quota" in low or "balance" in low or "no permission" in low):
                    raise RuntimeError(last)
                retriable = r.status_code in (429, 500, 502, 503, 504)
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            last = str(e)
            retriable = True
        if retriable and attempt < retries:
            wait = min(2 ** attempt * 2, 30)
            print(f"[qwen] 请求失败(第{attempt + 1}/{retries + 1}次): {last}，{wait}s 后重试")
            time.sleep(wait)
            continue
        raise RuntimeError(last or "qwen 调用失败")


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
    ap.add_argument("--only", default=None, help="只处理含该关键字的图片文件夹")
    ap.add_argument("--limit", type=int, default=0, help="最多处理前 N 张图（0=全部）")
    ap.add_argument("--max-solutions", type=int, default=0, help="每张只取前 N 个方案（0=全部）")
    ap.add_argument("--provider", default="qwen", choices=["openai", "gemini", "qwen"],
                    help="视觉评估后端：qwen(DashScope/qwen3.8-max 默认) / openai(gpt-4o-mini) / gemini(gemini-2.5-flash)")
    ap.add_argument("--model", default=None,
                    help="模型名（默认: qwen->qwen3.8-max, openai->gpt-4o-mini, gemini->gemini-2.5-flash）")
    ap.add_argument("--env", default=DEFAULT_ENV_PATH, help="含 OPENAI_API_KEY/GEMINI_API_KEY/QWEN_API_KEY 的 .env")
    ap.add_argument("--api-key", default=None, help="直接给 key（优先于 .env）")
    ap.add_argument("--proxy", default=None,
                    help="OpenAI 代理，如 http://127.0.0.1:30000（默认依次取 --proxy / "
                         "env OPENAI_PROXY / HTTPS_PROXY / HTTP_PROXY；本机需代理才能访问）")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max-side", type=int, default=900, help="原图压缩边长")
    ap.add_argument("--force", action="store_true", help="忽略缓存强制重评")
    args = ap.parse_args()

    key_var = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
               "qwen": "QWEN_API_KEY"}[args.provider]
    default_model = {"openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash",
                     "qwen": "qwen3.8-max"}[args.provider]
    model = args.model or default_model
    key = get_api_key(args.env, args.api_key, var=key_var)
    if not key:
        sys.exit(f"[qwen] 未找到 {key_var}（backend/.env 或 --api-key）")

    proxy = None
    if args.provider in ("openai", "gemini"):
        proxy = args.proxy or os.environ.get("OPENAI_PROXY") or os.environ.get("HTTPS_PROXY") \
            or os.environ.get("HTTP_PROXY") or "http://127.0.0.1:30000"
    print(f"[qwen] provider={args.provider} | model={model}" + (f" | 代理={proxy}" if proxy else ""))
    roots = list(args.data_root)
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
        # 回退根目录解析原图
        if img_path and not os.path.exists(img_path):
            for root in roots:
                cand = os.path.join(root, img_path.lstrip("/"))
                if os.path.exists(cand):
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
        if args.provider == "qwen":
            raw = call_qwen(key, model, messages)
        elif args.provider == "gemini":
            raw = call_gemini(key, model, messages, proxy=proxy)
        else:
            raw = call_gpt(key, model, messages, proxy=proxy)
        verdicts = normalize_verdicts(raw, len(sol_items))
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
