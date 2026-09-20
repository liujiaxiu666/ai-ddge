# -*- coding: utf-8 -*-
"""
DDGE 编码器封装：本地 qwen3-vl-8b-instruct，encoder-only 取 patch 多向量集合。

核心函数：get_dim_condition_patch_embedding(pil_img, condition_text)
  -> 返回该维度下的 patch 多向量集合 shape=[num_patch, hidden_dim]，L2 归一化。

⚠️ encoder-only：不调用 model.generate，读取 last hidden-state，
   用 image_mask（或 vision_start/end + image_grid_thw 回退法）过滤出视觉 patch token。
   编码器与生成器共用同一份 qwen3-vl-8b-instruct 权重（省显存）。
"""

from __future__ import annotations

import os
import sys

# 保证从 7dimensions 目录运行时能 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# OpenBLAS "too many memory regions" 段错误修复：
# 多线程 BLAS 在连续多次 forward / 共享内存池下会耗尽内存区域而崩溃（多次 "Program is Terminated"）。
# 必须在 import torch 之前锁定线程数；server.py 同样有该设置，这里兜底覆盖所有入口。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "8"))
os.environ.setdefault("MKL_NUM_THREADS", "1")

from config import CUDA_VISIBLE_DEVICES  # noqa: E402


# ⚠️ 必须在 import torch 之前设置 CUDA_DEVICE_ORDER + CUDA_VISIBLE_DEVICES：
#    1) CUDA 默认按 FASTEST_FIRST 枚举设备，序号与 nvidia-smi 的编号不一致
#       （本机 8 卡：CUDA 0-3 = RTX 5880 Ada(sm_89)，CUDA 4-7 = RTX PRO 5000 Blackwell(sm_120)）。
#       不锁定顺序时，.env 里按 nvidia-smi 写的 "7"（Ada）会落到 CUDA 的 7 号卡 = Blackwell，
#       torch cu126 没有 sm_120 内核 -> "CUDA error: no kernel image is available for execution"。
#       锁定 PCI_BUS_ID 后序号与 nvidia-smi 完全一致，选卡所见即所得。
#    2) torch 一旦初始化 CUDA，这两个变量再改就无效了；
#       CUDA_VISIBLE_DEVICES 不早设还会让 torch 看到全部 GPU（device_map 会 offload 到 CPU）。
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _cc_pair(cc: str) -> tuple:
    """算力字符串 -> (major, minor)。'12.0'/'sm_90a'/'8.6' 都能解析；失败返回 (-1, -1)。"""
    digits = "".join(ch for ch in str(cc) if ch.isdigit())
    if not digits:
        return (-1, -1)
    return (int(digits[:-1]), int(digits[-1])) if len(digits) >= 2 else (int(digits), 0)


def _caps_support(caps: set, cc: tuple) -> bool:
    """当前 torch 能否在该算力上跑：CUDA 同大版本内向前二进制兼容
    （sm_86 的 cubin 能在 sm_89 上跑），所以 major 相同且 minor 不低于即可。"""
    return any(c[0] == cc[0] and c[1] <= cc[1] for c in caps)


def _torch_supported_caps() -> set:
    """当前 torch 编译进去的算力集合（如 {(5,0)...(9,0)}）；失败返回空集合 = 不限制。
    只读 torch.cuda.get_arch_list()，不初始化 CUDA。"""
    try:
        import torch
        return {_cc_pair(a[3:]) for a in torch.cuda.get_arch_list() if a.startswith("sm_")}
    except Exception:  # noqa: BLE001
        return set()


def _auto_pick_gpu() -> str:
    """用 nvidia-smi 选当前显存最空闲的 GPU，跳过当前 torch 跑不了的算力（如 sm_120 + cu126）；
    失败回退 '0'。返回的索引与 nvidia-smi 一致（前面已锁定 CUDA_DEVICE_ORDER=PCI_BUS_ID）。"""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        rows = []
        for ln in out.stdout.strip().splitlines():
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                rows.append((parts[0], int(parts[1]), _cc_pair(parts[2])))
        if not rows:
            return "0"
        caps = _torch_supported_caps()
        usable = [r for r in rows if _caps_support(caps, r[2])]
        if usable:
            return max(usable, key=lambda r: r[1])[0]
        print(f"[encoder] ⚠️ 所有卡算力 {sorted({r[2] for r in rows})} 都不被当前 torch 支持"
              f"（支持 {sorted(caps)}），仍选显存最大的一张；请换卡或安装匹配 CUDA 的 torch",
              flush=True)
        return max(rows, key=lambda r: r[1])[0]
    except Exception:  # noqa: BLE001
        return "0"


if "CUDA_VISIBLE_DEVICES" in os.environ:
    _gpus = os.environ["CUDA_VISIBLE_DEVICES"].strip()
    if "," in _gpus:
        # 多卡列表（如 "0,1"）：单进程推理只用第一张；多进程并行时由各 worker 进程自己设单卡 env
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpus.split(",")[0].strip()
    # 单卡数字：尊重外部
elif CUDA_VISIBLE_DEVICES.strip().lower() in ("auto", ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = _auto_pick_gpu()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
print(f"[encoder] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}"
      f"（按 nvidia-smi 编号）")

import threading
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

import torch
import torch.nn.functional as F


def _check_device_supported() -> None:
    """可见卡算力不在当前 torch 支持列表时，给出比 torch 原生警告更可操作的提示
    （否则前向只报 'CUDA error: no kernel image is available for execution on the device'）。"""
    try:
        if not torch.cuda.is_available():
            return
        cc = tuple(torch.cuda.get_device_capability(0))
        caps = _torch_supported_caps()
        if caps and not _caps_support(caps, cc):
            print(f"[encoder] ❌ 当前可见卡 {torch.cuda.get_device_name(0)} 算力 sm_{cc[0]}{cc[1]} "
                  f"不在 torch {torch.__version__}(cuda {torch.version.cuda}) 支持列表 "
                  f"{sorted('sm_%d%d' % c for c in caps)} 内，前向会报 'no kernel image is available'。\n"
                  f"[encoder]    解决：① Blackwell(sm_120) 需安装 CUDA 12.8+ 对应的 torch; "
                  f"② 或改 .env 的 CUDA_VISIBLE_DEVICES 指向受支持的卡（编号同 nvidia-smi，"
                  f"本机 Ada(sm_89) 卡号为 4,5,6,7）。", flush=True)
    except Exception:  # noqa: BLE001
        pass


_check_device_supported()

# ---------------------------------------------------------------------------
# torchaudio ABI 兼容保险（升级 torch 后常见）：
#   torch 升到 2.12+（如 2.12.1+cu130）后，旧 torchaudio（如 2.7.0+cu128）会因
#   C++ 符号不匹配而导入失败（OSError: undefined symbol ... libtorchaudio.so）；
#   而 transformers 的 audio_utils 会无条件 `import torchaudio`，导致整条视觉链路
#   （本工程只做图像/文本，不用音频）连带 import 失败。
#   官方 torchaudio 已停止发版（PyPI 最高 2.11.0，无 2.12+ 配对版本），故这里在
#   导入 transformers 之前探测真实 torchaudio，不可用时注入轻量 stub。
#   若将来需要音频功能：改用 torchcodec / soundfile，或把 torch 降到与 torchaudio 匹配。
# ---------------------------------------------------------------------------
try:  # noqa: SIM105
    import torchaudio  # noqa: F401
except Exception as _ta_err:  # noqa: BLE001
    import sys as _sys
    import types as _types
    import importlib.machinery as _machinery
    _ta_stub = _types.ModuleType("torchaudio")
    _ta_stub.__version__ = "0.0.0-stub"
    # 让 importlib.util.find_spec("torchaudio") 正常返回（transformers 会调用它做
    # is_torchaudio_available 判断），否则会抛 ValueError: torchaudio.__spec__ is None
    _ta_stub.__spec__ = _machinery.ModuleSpec("torchaudio", loader=None)
    _ta_stub.__loader__ = None

    def _ta_missing(name):  # noqa: ANN001
        raise AttributeError(
            f"torchaudio 当前不可用（stub 模块），无法访问 {name}；"
            "本工程不使用音频功能，如需音频请安装与 torch 匹配的 torchaudio")

    _ta_stub.__getattr__ = _ta_missing
    _sys.modules["torchaudio"] = _ta_stub
    print(f"[encoder] torchaudio 不可用（{type(_ta_err).__name__}: {_ta_err}），"
          "已注入 stub（本工程不使用音频功能）", flush=True)

# 可用 VL 生成类注册表（按权重 config.json 的 model_type 精确匹配，
# 避免 Qwen3-VL 优先命中后拿错类去加载 Qwen2-VL 等不同结构权重）
try:
    from transformers import Qwen3VLForConditionalGeneration as _Qwen3VL
except Exception:  # noqa: BLE001
    _Qwen3VL = None
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as _Qwen25VL
except Exception:  # noqa: BLE001
    _Qwen25VL = None
try:
    from transformers import Qwen2VLForConditionalGeneration as _Qwen2VL
except Exception:  # noqa: BLE001
    _Qwen2VL = None

from transformers import AutoProcessor


def _pick_vl_class(model_path: str):
    """按权重目录 config.json 的 model_type 选正确的 VL 生成类；
    未知 model_type 时按可用类顺序回退（Qwen3-VL > Qwen2.5-VL > Qwen2-VL）。"""
    import json
    try:
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            mt = json.load(f).get("model_type", "")
    except Exception:  # noqa: BLE001
        mt = ""
    table = {"qwen3_vl": _Qwen3VL, "qwen2_5_vl": _Qwen25VL, "qwen2_vl": _Qwen2VL}
    if mt in table and table[mt] is not None:
        return table[mt]
    for cls in (_Qwen3VL, _Qwen25VL, _Qwen2VL):
        if cls is not None:
            return cls
    return None

from config import (DEVICE, DIM_CONDITION_PROMPTS,  # noqa: E402
                    DIM_ORDER, DTYPE, ENCODE_MAX_CONCURRENCY, ENCODE_SHARED_VISION,
                    ENCODE_USE_BATCH, ENCODER_MODEL_PATH, IMAGE_MAX_PIXELS, PATCH_POOL_K)

# ---------------------------------------------------------------------------
# 多模型共享缓存（编码器 + 生成器可同时常驻，文件夹批处理不重复加载）
# ---------------------------------------------------------------------------
_LOAD_LOCK = threading.Lock()
# 缓存上限：编码器(如 4B) + 生成器(如 2B) 两个正好；超过时按 LRU 淘汰最久未用的，防显存无限增长
MAX_CACHED_MODELS = int(os.getenv("DDGE_MAX_CACHED_MODELS", "2"))
_SHARED_CACHE = {}     # path -> {"model": ..., "processor": ...}
_SHARED_ORDER = []     # 按最近使用排序的 path 列表（最旧在前）


def _cache_touch(path: str) -> None:
    """标记 path 最近被使用（LRU 辅助）。"""
    if path in _SHARED_ORDER:
        _SHARED_ORDER.remove(path)
    _SHARED_ORDER.append(path)
    while len(_SHARED_ORDER) > MAX_CACHED_MODELS:
        _SHARED_ORDER.pop(0)


def get_shared_vlm(model_path: str = ENCODER_MODEL_PATH, dtype: str = DTYPE, device: str = DEVICE):
    """加载并缓存 VL 模型（编码与生成共用缓存；不同模型按 path 各自常驻复用）。"""
    with _LOAD_LOCK:
        if model_path in _SHARED_CACHE:
            entry = _SHARED_CACHE[model_path]
            _cache_touch(model_path)
            return entry["model"], entry["processor"]

        # 缓存满：LRU 淘汰最久未用的模型，腾出显存
        while len(_SHARED_CACHE) >= MAX_CACHED_MODELS:
            victim = _SHARED_ORDER.pop(0)
            _SHARED_CACHE.pop(victim, None)
            print(f"[encoder] LRU 淘汰缓存模型: {victim}（腾显存）", flush=True)
            torch.cuda.empty_cache()

        vl_cls = _pick_vl_class(model_path)
        if vl_cls is None:
            raise RuntimeError(
                "未找到可用的 VL 模型类（Qwen3VL / Qwen2.5-VL / Qwen2-VL），请检查 transformers 版本"
            )
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        # 单卡全加载（CUDA_VISIBLE_DEVICES 已在 import 前限制为单卡）：
        # 不用 device_map="auto"（多卡可见时会把层分散/offload 到 CPU，导致 device mismatch）
        print(f"[encoder] 使用模型类 {vl_cls.__name__} 加载: {model_path}", flush=True)
        model = vl_cls.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=True
        )
        model = model.to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # 限制输入图像最大像素，控制 patch 数量（越小 encode/入库越快）
        try:
            img_proc = getattr(processor, "image_processor", None)
            if img_proc is not None and hasattr(img_proc, "max_pixels"):
                img_proc.max_pixels = IMAGE_MAX_PIXELS
                print(f"[encoder] image max_pixels -> {IMAGE_MAX_PIXELS}")
        except Exception as e:  # noqa: BLE001
            print(f"[encoder] 设置 max_pixels 失败({e})，使用模型默认值")
        _SHARED_CACHE[model_path] = {"model": model, "processor": processor}
        _cache_touch(model_path)
        print(f"[encoder] 模型加载完成并缓存: {model_path}", flush=True)
        return model, processor


def _limit_pixels(img, max_pixels: int = IMAGE_MAX_PIXELS):
    """把超大图（如 48MP 广角）主动缩放到 max_pixels 内，避免 processor 处理极慢/显存爆炸。
    普通小图（≤ max_pixels）原样返回。"""
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)


class DdgeEncoder:
    """DDGE 编码器：输入 图片 + 维度条件文本 -> patch 多向量集合。"""

    def __init__(self, model_path: str = ENCODER_MODEL_PATH, dtype: str = DTYPE, device: str = DEVICE):
        self.model, self.processor = get_shared_vlm(model_path, dtype, device)
        self.device = device
        self.hidden_dim = self._resolve_hidden_dim()
        self.pool_k = PATCH_POOL_K   # 每维度 patch 集合池化成 K 个向量

    def _resolve_hidden_dim(self) -> int:
        """解析 patch 向量维度（last hidden-state 的 D）。
        - Qwen2.5-VL：顶层 config.hidden_size
        - Qwen3-VL  ：顶层 hidden_size/hidden_dim 均为 None，
                      真实维度在 text_config.hidden_size（文本塔，本机 4096）
        """
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            for attr in ("hidden_size", "hidden_dim"):
                v = getattr(cfg, attr, None)
                if isinstance(v, int) and v > 0:
                    return v
            # 嵌套子配置（Qwen3-VL 等统一 VL 模型）
            for sub in ("text_config", "language_config", "llm_config", "model_config"):
                sub_cfg = getattr(cfg, sub, None)
                if sub_cfg is not None:
                    v = getattr(sub_cfg, "hidden_size", None)
                    if isinstance(v, int) and v > 0:
                        return v
        print("[encoder] 未从 config 解析到 hidden_dim，回退到 2560")
        return 2560

    # ------------------------------------------------------------------
    # patch 池化：把 [N, D] 池化成 [K, D]（K=PATCH_POOL_K）
    # 只减少入库行数/索引体积，不减少模型前向时间；仍保留多向量集合，MaxSim 逻辑不变
    # ------------------------------------------------------------------
    def _pool_patch_set(self, patch_hs: torch.Tensor) -> torch.Tensor:
        n, _d = patch_hs.shape
        k = self.pool_k
        if k is None or k <= 0 or k >= n:
            return patch_hs
        seg = (n + k - 1) // k                      # 每组约 n/k 个 patch
        pooled = [patch_hs[i:i + seg].mean(dim=0) for i in range(0, n, seg)]
        return torch.stack(pooled)                  # [<=K, D]

    # ------------------------------------------------------------------
    # 核心：单张图片 + 单条条件提示 -> 该维度 patch 多向量集合 [N, D]
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_dim_condition_patch_embedding(self, pil_img, condition_text: str) -> torch.Tensor:
        messages = [{
            "role": "user",
            "content": [
                # ⚠️ 条件文本必须放在图片前面：Qwen-VL 是因果注意力，图片 token 只能看到其之前的 token；
                # 文本放图片后面则条件完全不影响 patch 向量（实测 7 维结果完全相同）
                {"type": "text", "text": condition_text},
                {"type": "image", "image": pil_img},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=[pil_img], return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        patch_hs = self._extract_patch_hidden(inputs, out)      # [N, D]
        return F.normalize(self._pool_patch_set(patch_hs), dim=-1)

    # ------------------------------------------------------------------
    # 并行：同一张图跑 7 次条件提示 -> dict[dim -> patch 集合]
    # 优先 batch 单次前向（快路径）；失败则回退多线程逐维前向。
    # ------------------------------------------------------------------
    def get_dim_patch_sets_parallel(self, pil_img, conditions=None, use_threads: bool = False) -> dict:
        conditions = conditions or {d: DIM_CONDITION_PROMPTS[d] for d in DIM_ORDER}
        # 大图先主动降到 max_pixels 内（否则 48MP 广角图编码极慢）
        pil_img = _limit_pixels(pil_img, IMAGE_MAX_PIXELS)
        # 方案 B（ENCODE_SHARED_VISION）：视觉塔只跑 1 次 + 一次 batch LLM（见 _batch_forward_shared_vision）
        if ENCODE_SHARED_VISION and not use_threads:
            try:
                return self._batch_forward_shared_vision(pil_img, conditions)
            except Exception as e:  # noqa: BLE001
                print(f"[encoder] 共享视觉塔前向失败({e})，回退常规路径", flush=True)
        # 默认走逐维编码（串行并发 ENCODE_MAX_CONCURRENCY）：显存水位低，单卡连续多张稳定；
        # batch 前向虽快但 7 条件一次前向显存峰值最高（默认关，可 ENCODE_USE_BATCH=True 开启）
        if ENCODE_USE_BATCH and not use_threads:
            try:
                return self._batch_forward(pil_img, conditions)
            except Exception as e:  # noqa: BLE001
                print(f"[encoder] batch 前向失败({e})，回退多线程逐维编码", flush=True)
        return self._thread_forward(pil_img, conditions)

    # ---- 快路径：7 条消息拼成 1 个 batch，单次前向并行出 7 套集合 ----
    @torch.no_grad()
    def _batch_forward(self, pil_img, conditions: dict) -> dict:
        items = list(conditions.items())                        # [(dim, cond), ...]
        texts, imgs = [], []
        for _dim, cond in items:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": cond},
                    {"type": "image", "image": pil_img},
                ],
            }]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            imgs.append(pil_img)

        inputs = self.processor(text=texts, images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = out.hidden_states[-1]                     # [B, L, D]

        results = {}
        for i, (dim, _cond) in enumerate(items):
            mask = self._image_mask_for_batch(inputs, i)        # [L] bool
            patch_hs = last_hidden[i][mask]
            results[dim] = F.normalize(self._pool_patch_set(patch_hs), dim=-1)
        return results

    # ------------------------------------------------------------------
    # 方案 B：共享视觉塔 batch 前向。
    # Qwen3-VL(DeepStack) 的视觉塔输出（pooler_output + deepstack_features）与文本无关，
    # 因此整张图只需过视觉塔 1 次，缓存后平铺到 batch 行，再做一次 batch LLM 前向即可。
    # 结果与「逐维单图全前向」逐位一致（同一 kernel 形状），比 _batch_forward（视觉算 7 遍）
    # 省约 6/7 视觉塔开销（实测端到端 ~23%）。
    # ⚠️ 需直连 HF 内部子模块：lm.language_model / get_image_features / get_placeholder_mask /
    #    compute_3d_position_ids；transformers 大版本升级可能失效（本机 5.3.0 已验证逐位等价）。
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _batch_forward_shared_vision(self, pil_img, conditions: dict) -> dict:
        model = self.model                       # Qwen3VLForConditionalGeneration
        lm = getattr(model, "model", None)       # Qwen3VLModel(.visual/.language_model/...)
        if lm is None or not all(hasattr(lm, a) for a in
                                 ("language_model", "get_image_features",
                                  "get_placeholder_mask", "compute_3d_position_ids")):
            raise RuntimeError("当前模型不支持共享视觉塔路径（无 lm.language_model 等子模块）")
        items = list(conditions.items())         # [(dim, cond), ...]
        texts, imgs = [], []
        for _dim, cond in items:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": cond},
                    {"type": "image", "image": pil_img},
                ],
            }]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            imgs.append(pil_img)

        inputs = self.processor(text=texts, images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # 1) 视觉塔只跑 1 次。⚠️ Qwen3-VL 的 pixel_values 是展平 patch 张量 [总patch数, C*p²*t]，
        #    不是 [B,C,H,W]：需按 每图像patch行 = 总行/B 切出第一张图，再配合 image_grid_thw[0:1]。
        B = len(items)
        total_px = inputs["pixel_values"].shape[0]
        per_img = total_px // B
        vout = lm.get_image_features(
            pixel_values=inputs["pixel_values"][:per_img],
            image_grid_thw=inputs["image_grid_thw"][0:1],
            return_dict=True,
        )
        single_emb = torch.cat(vout.pooler_output, dim=0)     # [N, D]
        single_deep = vout.deepstack_features                 # list of [N, D]
        img_embeds = single_emb.repeat(B, 1)                  # [B*N, D]
        deep_embeds = [d.repeat(B, 1) for d in single_deep]

        # 2) 复刻 HF Qwen3VLModel.forward 的 LLM 段：文本 embed + 视觉 embed scatter 进 image_pad
        input_ids = inputs["input_ids"]                       # [B, L]
        attn = inputs.get("attention_mask")
        inputs_embeds = lm.get_input_embeddings()(input_ids)
        image_mask, _ = lm.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=img_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, img_embeds.to(inputs_embeds.dtype))
        visual_pos_masks = image_mask[..., 0].bool()          # [B, L]
        position_ids = lm.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=None,
            attention_mask=attn,
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

        # 3) 一次 batch LLM 前向（deepstack 特征在前 3 层注入视觉位置）
        out = lm.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attn,
            cache_position=None,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deep_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1]                   # [B, L, D]

        results = {}
        for i, (dim, _cond) in enumerate(items):
            mask = self._image_mask_for_batch(inputs, i)      # [L] bool
            patch_hs = last_hidden[i][mask]
            results[dim] = F.normalize(self._pool_patch_set(patch_hs), dim=-1)
        return results

    # ---- 回退：多线程逐维编码 ----
    def _thread_forward(self, pil_img, conditions: dict) -> dict:
        def work(dim):
            return dim, self.get_dim_condition_patch_embedding(pil_img, conditions[dim])
        # 限制并发数，控制显存峰值（见 config.ENCODE_MAX_CONCURRENCY）
        workers = min(ENCODE_MAX_CONCURRENCY, len(conditions))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return dict(ex.map(work, list(conditions.keys())))

    # ------------------------------------------------------------------
    # patch token 过滤：Qwen3-VL / Qwen2.5-VL 通用
    # 模型输出的 last_hidden 形状 [B, L, D]，整条序列为：
    #   文本token + <|vision_start|> + 图片patch token + <|vision_end|> + 后续文本token
    # 三种定位方式（按优先级）：
    #   1. image_mask（新版 transformers 的 Qwen2.5-VL 会返回）
    #   2. <|image_pad|> token 位置（Qwen3-VL 特有：图像被 2x2 合并后，
    #      每个 <|image_pad|> 对应 hidden 里一个视觉 patch 行）
    #   3. <|vision_start|>/<|vision_end|> 之间的区间（通用回退）
    # ------------------------------------------------------------------
    def _build_patch_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """构建与 hidden 序列对齐的视觉 patch 位置布尔掩码（[L] bool）。"""
        tok = self.processor.tokenizer

        # Qwen3-VL：<|image_pad|> 逐 token 代表每个合并后的视觉 patch
        pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
        if isinstance(pad_id, int) and pad_id >= 0 and (input_ids == pad_id).any():
            return input_ids == pad_id

        # 通用回退：<|vision_start|> .. <|vision_end|> 之间的行（Qwen2.5-VL 等）
        sid = tok.convert_tokens_to_ids("<|vision_start|>")
        eid = tok.convert_tokens_to_ids("<|vision_end|>")
        start = (input_ids == sid).nonzero()
        end = (input_ids == eid).nonzero()
        if start.numel() and end.numel():
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            mask[start[0].item() + 1:end[0].item()] = True
            return mask
        raise RuntimeError("无法定位视觉 patch token（无 image_mask / image_pad / vision_start/end）")

    def _image_mask_for_batch(self, inputs, i: int) -> torch.Tensor:
        if "image_mask" in inputs:
            mask = inputs["image_mask"]                         # [B, L] 或 [B,1,L]
            if mask.ndim == 3:
                mask = mask[:, 0, :]
            return mask[i].bool()
        # Qwen3-VL 等无 image_mask 的模型：用 input_ids 里的视觉 token 定位
        return self._build_patch_mask(inputs["input_ids"][i])

    def _extract_patch_hidden(self, inputs, out) -> torch.Tensor:
        last_hidden = out.hidden_states[-1]                     # [B, L, D]
        ids = inputs["input_ids"][0]                            # [L]
        if "image_mask" in inputs:
            mask = inputs["image_mask"]
            if mask.ndim == 3:
                mask = mask[:, 0, :]
            mask = mask[0].bool()
        else:
            mask = self._build_patch_mask(ids)
        return last_hidden[0][mask]                             # [N, D]
