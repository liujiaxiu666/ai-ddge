# -*- coding: utf-8 -*-
"""
拍照指导生成模块。

把按维度分组的检索证据送入 qwen3-vl-8b-instruct（本地）生成拍照指导。
硬性约束：**某维度的拍照整改建议，只能使用本组维度的参考图片与文本证据，
禁止跨维度挪用证据**。

检索证据按维度对齐：
  evidence_by_dim = {
    "ratio":       [{sample_id, orig_image_path(好图), control_image_path(坏图), dim_original_text, score}, ...],
    "composition": [...],
    ...
  }
**每份方案每个维度只取第 k 张候选好图（1 张图 + 对应 1 条文本）**拼进 prompt 作为该维参考，
绝不发送该维全部文本；是否把该维「好图」参考图送入模型由 config.INCLUDE_DIM_GOOD_IMAGE 控制。

输出：生成若干份方案并写入 output/<图片名>/solutions.json，包含 evidence_by_dim、
solutions、edit_images 与 timing_s{retrieval, vlm_generation, image_edit, total}。
方案数动态决定（不再硬编码）：方案 k = 每个维度取检索到的第 k 张候选好图 + 该维对应文本
（实际方案数 = 各维候选好图数的最小值，检索时每维至少凑 MIN_SOLUTIONS 张候选好图）。
qwen(API) 方案并行生成走多 key 并发池（config.QWEN_API_KEYS_CONCURRENCY，默认 2/2/1=5 并发）。
payload 的 mode 字段标识调用来源：
  - retrieve_plus_prompt                 （CLI: python main.py infer ...）
  - retrieve_plus_prompt_plus_image_edit （CLI: python main.py infer ... --image-edit）
只检索（不调本模块生成）走 user.py 的 generate_prompt=False，输出 retrieval_summary.json。
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time

import requests
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import queue
import threading

from concurrent.futures import Future

from transformers import TextIteratorStreamer

from config import (DIM_CN, DIM_ORDER, DTYPE,                            # noqa: E402
                    GEN_TEMPERATURE, GENERATE_MAX_NEW_TOKENS,
                    GENERATE_PARALLEL,
                    IMAGE_EDIT_ENABLED, IMAGE_EDIT_MODEL,
                    IMAGE_EDIT_PARALLEL, IMAGE_EDIT_PROMPT_SECTION,
                    INCLUDE_DIM_GOOD_IMAGE,
                    MAX_SOLUTIONS, N_SOLUTIONS, OUTPUT_DIR,
                    QWEN_API_KEY, QWEN_API_KEYS_CONCURRENCY, QWEN_BASE_URL,
                    VLM_API_MAX_RETRIES,
                    parse_model_spec, resolve_local_model,
                    VLLM_BASE_URL, VLM_GENERATOR_MODEL, VLM_MODEL_NAME,
                    VLM_PROVIDER)
from encoder import get_shared_vlm                                       # noqa: E402


def _pil_to_b64(img) -> str:
    """PIL 图片 -> data:image/jpeg;base64（API 模式用）；先降采样到 1MP 内，
    避免超大原图（如 48MP 广角）使 base64 请求体过大导致 API 400。"""
    img = img.convert("RGB")
    try:
        from encoder import _limit_pixels
        img = _limit_pixels(img, 1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode()


def _extract_advice_section(text: str) -> str:
    """从方案文本中截取“# 针对性整改建议”到结尾的部分（图编 prompt 用）。
    找不到该标题时回退为整份文本。
    """
    for marker in ("# 针对性整改建议", "# 针对性整改"):
        idx = text.find(marker)
        if idx != -1:
            return text[idx:]
    return text

# 硬性生成约束（保持原注释不变，只做格式整理）
CONSTRAINTS = (
    "====硬性生成约束====\n"
    "1. 分析用户照片7个维度（比例、构图、机位、主体位置、姿态、对焦、色彩）分别存在什么问题。\n"
    "2. 针对【Ratio画面比例】输出的整改建议，只能使用【Ratio画面比例证据组】里面图片与文本，禁止使用构图/色彩等其他组证据。\n"
    "3. 其余维度同理，每个维度整改建议仅允许使用本组证据。\n"
    "4. 只要某维度提供了参考证据，就必须基于该组证据写出该维度的具体整改建议，严禁写“暂无该维度相关参考证据”；\n"
    "5. 仅当某维度确实没有检索到任何参考证据（未提供该维证据）时，才允许写“暂无该维度相关参考证据”，严禁编造摄影知识。\n\n"
    "输出格式：\n"
    "# 拍照缺陷分析\n"
    "- 画面比例：xxx\n"
    "- 构图取景：xxx\n"
    "……\n\n"
    "# 针对性拍照整改指导\n"
    "- 画面比例整改：xxx\n"
    "- 构图取景整改：xxx\n"
    "……"
)


class QwenApiPool:
    """qwen(DashScope) API 多 key 并发池：用多个 key 分摊单 key 限流（429）/余额不足。

    按 config.QWEN_API_KEYS_CONCURRENCY（默认 key1:2 / key2:2 / key3:1 = 共 5 并发）为
    每个 key 起固定数量的常驻工作线程（= 该 key 的并发槽数）。任务经队列分发给各 key
    的线程执行：既限制总并发（= 总槽数，5），又保证每个 key 的并发不超过其槽位。
    **某个 key 失败（余额不足 / key 失效 / 429 等）时，任务自动换剩余 key 重试，绝不跳过**；
    只有所有 key 都失败才返回异常。空 key 自动跳过。工作线程常驻，多图 / 常驻服务复用同一池。
    """

    def __init__(self, keys_conc=None):
        if keys_conc is None:
            keys_conc = QWEN_API_KEYS_CONCURRENCY
        self._queue = queue.Queue()
        self._n_workers = 0
        self._keys = []
        for key, c in keys_conc:
            if not key:
                continue
            self._keys.append(key)
            for _ in range(max(1, int(c))):
                threading.Thread(target=self._worker, args=(key,), daemon=True).start()
                self._n_workers += 1
        if not self._n_workers:
            # 兜底：没有任何 key 时也留 1 个槽（API 会报 401，但流程不挂）
            self._keys = [""]
            threading.Thread(target=self._worker, args=("",), daemon=True).start()
            self._n_workers = 1
        print(f"[prompt] qwen 并发池就绪：{self._keys} -> 共 {self._n_workers} 并发槽"
              f"（单 key 失败自动换 key 重试）", flush=True)

    def _worker(self, key: str):
        while True:
            fut, fn, tried = self._queue.get()
            if key in tried:
                # 该 key 已在本任务上失败过：放回队尾，交给其他 key 的槽重试（避免原地空转）
                self._queue.put((fut, fn, tried))
                time.sleep(0.005)
                continue
            try:
                fut.set_result(fn(key))
            except BaseException as e:  # noqa: BLE001
                new_tried = tried + [key]
                if len(new_tried) < len(self._keys):
                    # 还有别的 key 没试过：换 key 重试（不跳过这张图）
                    print(f"[prompt] qwen key 失败（{key[:6]}...），换剩余 key 重试: {e}",
                          flush=True)
                    self._queue.put((fut, fn, new_tried))
                else:
                    print(f"[prompt] 所有 qwen key 均失败，该方案失败: {e}", flush=True)
                    fut.set_exception(e)

    def submit(self, fn):
        """fn(api_key) -> result；立即返回 Future，由池内线程执行（流式拉取在池线程内完成）。
        单个 key 失败会自动用剩余 key 重试（不跳过）；所有 key 都失败才返回异常。"""
        fut = Future()
        self._queue.put((fut, fn, []))
        return fut


_pool_lock = threading.Lock()
_pool_singleton: QwenApiPool | None = None


def get_qwen_api_pool() -> QwenApiPool:
    """全局单例：多图 / 常驻服务共享同一批工作线程（避免每图重建线程）。"""
    global _pool_singleton
    if _pool_singleton is None:
        with _pool_lock:
            if _pool_singleton is None:
                _pool_singleton = QwenApiPool()
    return _pool_singleton


class GuidanceGenerator:
    """基于 qwen3-vl-8b-instruct 生成拍照指导（与编码器共用同一份权重）。"""

    def __init__(self, model_spec: str = VLM_GENERATOR_MODEL, dtype: str = DTYPE):
        """provider: qwen(远程API) / vllm(本地vLLM) / local(本地transformers)。"""
        if model_spec == VLM_GENERATOR_MODEL:
            provider, name = VLM_PROVIDER, VLM_MODEL_NAME
        else:
            # 统一走 config 解析：支持绝对路径 / local/ 前缀 / provider/model 三种写法
            provider, name = parse_model_spec(model_spec)
        self.provider, self.model_spec = provider, model_spec
        self.device = None
        self.model = self.processor = None
        self.api_base_url = self.api_model = self.api_key = None
        if provider in ("qwen", "api"):
            self.api_base_url, self.api_model, self.api_key = QWEN_BASE_URL, name, QWEN_API_KEY
        elif provider in ("vllm", "vllm-local"):
            self.api_base_url, self.api_model, self.api_key = VLLM_BASE_URL, name, ""
        else:  # local
            self.model, self.processor = get_shared_vlm(resolve_local_model(name), dtype)
            self.device = next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # 组装「单份方案」的 chat 消息（含图片 + 本方案参考证据：每维仅 1 张候选好图 + 1 条文本）
    # ------------------------------------------------------------------
    def build_solution_messages(self, user_img, evidence_by_dim: dict,
                                solution_idx: int, n_solutions: int):
        """返回 chat messages：
        - 第 1 张图：用户实拍图片
        - 每个维度组：只取该维第 solution_idx 张候选好图（**1 张图 + 1 条对应文本**）
        - 本方案标识（方案 solution_idx / 共 n_solutions）+ 输出格式
        硬性约束：每个方案每个维度**只**用 1 张好图 + 对应 1 条文本，绝不发送该维全部文本。
        """
        content = []

        # ---- 用户实拍图片 ----
        content.append({"type": "text", "text": f"用户实拍图片（第 {solution_idx}/{n_solutions} 个方案）："})
        content.append({"type": "image", "image": user_img})
        content.append({"type": "text", "text": (
            "你是图片审美诊断助手。基于用户原图与分组参考证据，输出一份拍照整改方案。\n"
            f"本方案参考证据：每个维度检索到的第 {solution_idx} 张候选好图及其文本（方案 {solution_idx}）。\n"
            "要求：\n"
            "1. 先诊断用户照片 7 个维度（比例、构图、机位、主体位置、姿态、对焦、色彩）的具体问题；\n"
            "2. 对每个维度给出可落地的整改建议，且【只允许使用该维度本组证据】，禁止挪用其他组证据；\n"
            "3. 只要该维提供了参考证据，就必须基于本组证据写出该维度的具体整改建议，严禁写“暂无该维度相关参考证据”；仅当某维确实完全没有提供任何证据时才允许写“暂无”，严禁编造摄影知识；\n"
            "4. 参考图/参考文本是正确示范，借鉴其审美处理方式，但必须针对用户照片的具体问题原创给出，不照搬参考场景。\n"
            "====分组参考证据（禁止跨分组挪用证据！）===="
        )})

        # ---- 按维度分组参考证据：每个维度【只】取第 solution_idx 张候选好图（1 图 + 1 文）----
        for dim in DIM_ORDER:
            evs = (evidence_by_dim or {}).get(dim, [])
            if solution_idx - 1 >= len(evs):
                content.append({"type": "text", "text":
                    f"\n【{DIM_CN[dim]}维度】\n暂无该维度相关参考证据"})
                continue

            ev = evs[solution_idx - 1]
            content.append({"type": "text", "text":
                f"\n【{DIM_CN[dim]}维度】\n[本方案参考：第 {solution_idx} 张候选好图（每维仅此 1 张）及其文本，必须基于该证据写出该维整改建议，严禁写“暂无该维度相关参考证据”]"})
            if INCLUDE_DIM_GOOD_IMAGE and ev.get("good_image_path"):
                try:
                    content.append({"type": "image",
                                    "image": Image.open(ev["good_image_path"]).convert("RGB")})
                except Exception:  # noqa: BLE001
                    pass
            content.append({"type": "text", "text":
                f"参考文本整改建议：{ev['dim_original_text']}"})

        content.append({"type": "text", "text": (
            "\n请输出：\n"
            f"# 方案{solution_idx} 拍照整改指导（参考每维度第 {solution_idx} 张候选好图）\n"
            "# 拍照缺陷分析\n"
            "- 画面比例：xxx\n- 构图取景：xxx\n……（7 个维度）\n\n"
            "# 针对性整改建议\n"
            "- 画面比例整改：xxx\n- 构图取景整改：xxx\n……"
        )})

        return [{"role": "user", "content": content}]

    # ------------------------------------------------------------------
    # 流式生成文本（yield token 块，供逐方案/流式输出）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_stream(self, messages, max_new_tokens: int = GENERATE_MAX_NEW_TOKENS,
                        temperature: float = GEN_TEMPERATURE, api_key: str | None = None):
        if self.provider in ("qwen", "api", "vllm", "vllm-local"):
            yield from self._api_stream(messages, max_new_tokens, temperature, api_key=api_key)
            return

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 收集 content 里所有图片（顺序即 <image> token 顺序）
        images = [item["image"] for item in messages[0]["content"]
                  if isinstance(item, dict) and item.get("type") == "image"]

        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # 生成前清缓存，释放编码阶段的显存，避免 24GB 卡 OOM
        torch.cuda.empty_cache()
        streamer = TextIteratorStreamer(self.processor.tokenizer,
                                        skip_prompt=True, skip_special_tokens=True)

        def _run():
            self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=True, temperature=temperature, streamer=streamer)

        threading.Thread(target=_run, daemon=True).start()
        for chunk in streamer:
            yield chunk

    # ------------------------------------------------------------------
    # 远程 DashScope qwen：OpenAI 兼容接口流式调用（requests 直连，无 SDK 依赖）
    # api_key 为空时用 self.api_key（多 key 并发池会把具体 key 传进来分摊限流）。
    # 对 429/5xx（限流/服务暂不可用）做指数退避重试，减少被限流打断。
    # ------------------------------------------------------------------
    def _api_stream(self, messages, max_new_tokens: int, temperature: float,
                    api_key: str | None = None, max_retries: int = VLM_API_MAX_RETRIES):
        key = api_key if api_key is not None else self.api_key
        payload_messages = []
        for msg in messages:
            content = []
            for item in msg["content"]:
                if item.get("type") == "image":
                    content.append({"type": "image_url",
                                    "image_url": {"url": _pil_to_b64(item["image"])}})
                else:
                    content.append({"type": "text", "text": item.get("text", "")})
            payload_messages.append({"role": msg.get("role", "user"), "content": content})

        url = f"{self.api_base_url}/chat/completions"
        body = {
            "model": self.api_model, "messages": payload_messages,
            "max_tokens": max_new_tokens, "temperature": temperature,
            "stream": True,
        }
        last_exc = None
        for attempt in range(max_retries + 1):
            resp = requests.post(url,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"},
                                 json=body, stream=True, timeout=300)
            if resp.status_code == 200:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        return
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                    except Exception:  # noqa: BLE001
                        continue
                    if delta:
                        yield delta
                return
            # 非 200：打印响应体便于定位（如图片数超限 / 上下文超长）
            print(f"[prompt] VLM API HTTP {resp.status_code}: {resp.text[:300]}", flush=True)
            # 仅对可重试的瞬时错误（限流 429 / 5xx）退避重试；400/401/403 等直接抛（并发池会换 key）
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = min(2 ** attempt * 2, 30)   # 退避 2s,4s,8s...（qwen 限流窗口较长）
                print(f"[prompt] 限流/服务暂不可用，{wait}s 后重试（{attempt + 1}/{max_retries}）",
                      flush=True)
                time.sleep(wait)
                last_exc = requests.HTTPError(f"HTTP {resp.status_code} (transient)")
                continue
            resp.raise_for_status()
        raise last_exc if last_exc is not None else RuntimeError("VLM API 重试耗尽")

    # ------------------------------------------------------------------
    # 生成若干份方案（每份=每维第 k 张候选好图；流式逐方案输出）+ 保存（每份 .md + 汇总 json）
    # ------------------------------------------------------------------
    def generate_solutions_and_save(self, user_img_path: str, evidence_by_dim: dict,
                                    output_dir: str = OUTPUT_DIR,
                                    n_solutions: int = N_SOLUTIONS,
                                    max_new_tokens: int = GENERATE_MAX_NEW_TOKENS,
                                    temperature: float = GEN_TEMPERATURE,
                                    retrieval_s: float = 0.0,
                                    encode_s: float = 0.0,
                                    faiss_search_s: float = 0.0,
                                    image_edit_async: bool = False,
                                    image_edit_enabled: bool | None = None):
        """生成若干份方案（流式逐方案）+ 可选图编，保存为 输出目录/<图片名>/ 文件夹：
        方案数动态决定（不再硬编码）：方案 k = 每个维度取第 k 张候选好图 + 该维文本。
        n = min(各维候选好图数)，并受 n_solutions(上限参数) / config.MAX_SOLUTIONS 约束。
        单个方案失败（如所有 qwen key 都失败）只记录 error 并继续，不会拖垮整张图；
        **永远写 solutions.json**（含已成功方案 + errors + 各阶段计时 s）。
        内含若干张编辑图 + solutions.json。返回: (solutions, json_path, edit_paths)
        """
        effective_image_edit_enabled = IMAGE_EDIT_ENABLED if image_edit_enabled is None else image_edit_enabled
        user_img = Image.open(user_img_path).convert("RGB")
        stem = os.path.splitext(os.path.basename(user_img_path))[0]
        out_dir = os.path.join(output_dir, stem)          # 以图片名命名的文件夹
        os.makedirs(out_dir, exist_ok=True)

        # ---- 0) 动态方案数：= 各维候选好图数的最小值（第 k 张好图存在，方案 k 才成立）----
        avail = min((len(evs) for evs in (evidence_by_dim or {}).values() if evs), default=0)
        if avail <= 0:
            avail = 1                 # 极端兜底：无任何维度证据时也至少出 1 份
        n = avail if n_solutions is None else min(avail, n_solutions)
        if MAX_SOLUTIONS:
            n = min(n, MAX_SOLUTIONS)
        if n <= 0:
            n = 1
        cap_note = "" if n_solutions is None else f"（上限 {n_solutions}）"
        print(f"[prompt] 各维候选好图数最小 {avail} 张 -> 生成 {n} 份方案{cap_note}", flush=True)

        # ---- 1) VLM 生成 n 份方案（每维只取第 k 张候选好图 + 对应文本；并行 + 计时） ----
        t_vlm0 = time.time()
        solutions = [None] * n

        def _build_solution(api_key, k):
            """方案 k（1-based）：每个维度只取第 k 张候选好图 + 该维文本（1 图 + 1 文）。"""
            label = f"参考每维度第 {k} 张候选好图"
            reference_images = {}
            for dim, evs in (evidence_by_dim or {}).items():
                if k - 1 < len(evs) and evs[k - 1].get("good_image_path"):
                    reference_images[dim] = evs[k - 1]["good_image_path"]
            messages = self.build_solution_messages(user_img, evidence_by_dim, k, n)
            chunks = list(self.generate_stream(messages, max_new_tokens=max_new_tokens,
                                               temperature=temperature, api_key=api_key))
            return k, {"index": k, "direction": label, "text": "".join(chunks).strip(),
                       "reference_images": reference_images}

        def _collect(k, sol):
            print(f"\n[prompt] ==== 方案 {k}/{n}（{sol['direction']}）完成 ====")
            print(sol["text"])
            solutions[k - 1] = sol

        errors = []

        def _record_failure(k, e):
            """某个方案失败（如所有 qwen key 都失败）：记录 error 并继续，不让单方案失败拖垮整张图。"""
            print(f"[prompt] 方案 {k}/{n} 生成失败（记录并继续）: {e}", flush=True)
            solutions[k - 1] = {"index": k, "direction": f"参考每维度第 {k} 张候选好图",
                                "text": "", "error": str(e)}
            errors.append({"solution": k, "error": str(e)})

        if self.provider in ("qwen", "api"):
            # qwen(API)：走多 key 并发池（key1×2 + key2×2 + key3×1 = 5 并发；GENERATE_PARALLEL 不生效）
            from concurrent.futures import as_completed
            pool = get_qwen_api_pool()
            if n > 1:
                futs = {k: pool.submit(lambda key, k=k: _build_solution(key, k))
                        for k in range(1, n + 1)}
                fut_to_k = {fut: k for k, fut in futs.items()}
                for fut in as_completed(futs.values()):
                    k = fut_to_k[fut]
                    try:
                        _, sol = fut.result()
                    except Exception as e:  # noqa: BLE001
                        _record_failure(k, e)
                        continue
                    _collect(k, sol)
            else:
                try:
                    _, sol = pool.submit(lambda key: _build_solution(key, 1)).result()
                except Exception as e:  # noqa: BLE001
                    _record_failure(1, e)
                else:
                    _collect(1, sol)
        elif GENERATE_PARALLEL and n > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = {ex.submit(_build_solution, None, k): k for k in range(1, n + 1)}
                for fut in as_completed(futs):
                    k = futs[fut]
                    try:
                        _, sol = fut.result()
                    except Exception as e:  # noqa: BLE001
                        _record_failure(k, e)
                        continue
                    _collect(k, sol)
        else:
            for k in range(1, n + 1):
                try:
                    _, sol = _build_solution(None, k)
                except Exception as e:  # noqa: BLE001
                    _record_failure(k, e)
                    continue
                _collect(k, sol)
        vlm_s = time.time() - t_vlm0

        # ---- 2) 可选：图编并行编辑用户照片（计时；支持后台异步，不阻塞返回） ----
        edit_paths = []
        edit_s = 0.0
        if effective_image_edit_enabled:
            prompts = []
            for s in solutions:
                if not s.get("text"):
                    continue          # 该方案生成失败，跳过图编
                body = s["text"]
                if IMAGE_EDIT_PROMPT_SECTION == "advice":
                    body = _extract_advice_section(body)   # 只取“# 针对性整改建议”段
                prompts.append(
                    "请对这张照片按以下方案进行后期编辑（保持主体与场景不变，只调整摄影审美因素）：\n"
                    f"【方案{s['index']}·{s['direction']}】\n{body}"
                )

            def _do_edit():
                """执行图编，返回 (edit_paths, elapsed_s)；异步时由线程调用并回写 json。"""
                from image_edit import edit_image_parallel, save_edit_image
                print(f"\n[prompt] 调用图编模型 {IMAGE_EDIT_MODEL} 并行编辑 {len(prompts)} 张 "
                      f"(取段={IMAGE_EDIT_PROMPT_SECTION}) ...", flush=True)
                t_e = time.time()
                try:
                    edited = edit_image_parallel(user_img, prompts,
                                                 parallel=IMAGE_EDIT_PARALLEL)
                except Exception as e:  # noqa: BLE001
                    # 图编失败（如 429 重试耗尽）不中断主流程，json 保持 edit_images=[]
                    print(f"[prompt] 图编失败（跳过，本次无编辑图）: {e}", flush=True)
                    return [], 0.0
                paths = []
                good = [s for s in solutions if s.get("text")]   # 只对生成成功的方案图编
                for i, (res, s) in enumerate(zip(edited, good), start=1):
                    ep = os.path.join(out_dir, f"solution_{i}.png")
                    save_edit_image(res, ep)
                    paths.append(ep)
                    s["edit_image"] = ep
                elapsed = time.time() - t_e
                # 回写 json：异步后台场景此时 solutions.json 已存在，更新 edit_images / 图编计时；
                # 同步场景文件尚未写出，这里读不到会静默跳过，由主流程第 3 步统一写正确值。
                try:
                    jp = os.path.join(out_dir, "solutions.json")
                    with open(jp, "r", encoding="utf-8") as f:
                        pl = json.load(f)
                    pl["edit_images"] = paths
                    pl["timing_s"]["image_edit"] = round(elapsed, 2)
                    pl["timing_s"]["total"] = round(pl["timing_s"]["retrieval"]
                                                     + pl["timing_s"]["vlm_generation"]
                                                     + pl["timing_s"]["image_edit"], 2)
                    with open(jp, "w", encoding="utf-8") as f:
                        json.dump(pl, f, ensure_ascii=False, indent=2)
                except Exception:  # noqa: BLE001
                    pass
                print(f"[prompt] 图编完成: {paths}", flush=True)
                return paths, elapsed

            if image_edit_async:
                import threading
                threading.Thread(target=_do_edit, daemon=True).start()
                print("[prompt] 图编已提交后台异步（响应不等待图编）", flush=True)
            else:
                edit_paths, edit_s = _do_edit()

        # ---- 3) 汇总 json（末尾含各阶段计时 s） ----
        total_s = retrieval_s + vlm_s + edit_s
        js_path = os.path.join(out_dir, "solutions.json")
        payload = {
            "input_image": user_img_path,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "retrieve_plus_prompt" if not effective_image_edit_enabled else "retrieve_plus_prompt_plus_image_edit",
            "model": self.model_spec,
            "vlm_generator_mode": self.provider,
            "evidence_by_dim": evidence_by_dim,
            "solutions": solutions,
            "errors": errors,
            "edit_images": (edit_paths
                             if not (effective_image_edit_enabled and image_edit_async)
                             else "processing"),
            "timing_s": {
                "retrieval": round(retrieval_s, 2),
                "encode": round(encode_s, 2),
                "faiss_search": round(faiss_search_s, 2),
                "vlm_generation": round(vlm_s, 2),
                "image_edit": round(edit_s, 2),
                "total": round(total_s, 2),
            },
        }
        with open(js_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"\n[prompt] 已保存到文件夹: {out_dir}（{len(edit_paths)} 张编辑图 + solutions.json）")
        print(f"[prompt] 计时(s): 检索 {payload['timing_s']['retrieval']} | "
              f"VLM生成 {payload['timing_s']['vlm_generation']} | "
              f"图编 {payload['timing_s']['image_edit']} | 总计 {payload['timing_s']['total']}")
        if errors:
            print(f"[prompt] 警告: {len(errors)}/{n} 个方案生成失败（已记录在 solutions.json 的 errors 字段）",
                  flush=True)
        return solutions, js_path, edit_paths


if __name__ == "__main__":
    # 单独调试：python prompt.py <user_img> <evidence.json>
    user_img_arg = sys.argv[1] if len(sys.argv) > 1 else "user_photo.jpg"
    ev_path = sys.argv[2] if len(sys.argv) > 2 else ""
    with open(ev_path, "r", encoding="utf-8") as f:
        evidence = json.load(f)
    gen = GuidanceGenerator()
    solutions, js, edits = gen.generate_solutions_and_save(user_img_arg, evidence)
    print("已保存:", js)
    print("编辑图:", edits)