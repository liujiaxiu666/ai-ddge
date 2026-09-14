# -*- coding: utf-8 -*-
"""
7 维度 DDGE（Dimension-Decomposed Generative Embedding）—— 拍照指导系统配置

整体 pipeline：
  1) AesRecon 每条样本 = 参考图片 + content；content 拆出 7 个独立维度文本片段：
     [Ratio, Composition, Camera, Position, Pose, Focus, Color]。
  2) 离线阶段：用「坏图」(control_path) 循环跑 7 次编码器（本地 qwen3-vl-8b-instruct，encoder-only），
     每次输入「坏图 + 对应维度条件提示」，输出 7 套独立 patch 多向量集合，
     每套 patch 集合绑定维度元数据（dim / sample_id / 好图路径 orig_image_path / 坏图路径 control_image_path / 该维度原始文本）存入 FAISS 索引。
  3) 推理阶段：用户图片（支持单张或整个文件夹）同样并行跑 7 次条件提示，
     得到 7 套查询 patch 集合；按维度对齐并行做 ANN + MaxSim（集合-集合晚交互）检索；
     检索结果按维度 + **好图作为参考样图**送入 prompt.py（qwen3-vl-8b-instruct）生成拍照指导并保存。
"""

import os

# ⚠️ CUDA 默认按 FASTEST_FIRST 枚举设备，序号与 nvidia-smi 顺序不一致（本机 8 卡时刚好相反：
#    CUDA 0-3 = Ada，CUDA 4-7 = Blackwell）。不锁定的话，.env / "auto" 里按 nvidia-smi 写的卡号
#    会落到别的卡上（曾选到 sm_120 的 Blackwell，torch cu126 无对应内核 -> "no kernel image"）。
#    必须在 import torch（首次 CUDA 调用）之前生效，所以放在 config 最前面做全局兜底。
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 0. 内部函数（全部集中在这里；下面只剩“配置项”，改参数一眼可见）
# ============================================================================
def _load_dotenv(*paths: str) -> None:
    """把 .env 里的 KEY=VALUE 读进环境变量（已存在的环境变量优先，不覆盖）。失败静默。"""
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except FileNotFoundError:
            continue


def _data_path(rel: str) -> str:
    """共享盘规范路径优先；本机挂载点不同（未挂载）时回退仓库内 dataset/ 软链。
    子目录尚不存在时也返回本地挂载点路径（避免落到本机并不存在的规范路径）。"""
    abs_p = os.path.join(_SHARED_DATA, rel)
    if os.path.exists(abs_p):
        return abs_p
    link_p = os.path.join(_REPO_DIR, "dataset", rel)
    if os.path.isdir(os.path.dirname(link_p)):
        return link_p
    return abs_p


def resolve_local_model(spec: str) -> str:
    """把 'local/xxx'、相对名或绝对路径解析为绝对权重路径。"""
    s = spec.strip().replace("\\", "/")
    s = os.path.expanduser(s)
    if s.startswith("local/"):
        s = s[len("local/"):]
    if os.path.isabs(s):
        return s
    return os.path.join(LOCAL_MODEL_ROOT, s)


def parse_model_spec(spec: str):
    """返回 (provider, model_name)。规则：
    - 绝对路径 / 'local/'       -> 本地 transformers
    - 'vllm/xxx'                -> 本地 vLLM
    - 裸模型名 'qwen3.8-flash'  -> DashScope 原生接口（可思考）；'qwen38/xxx' 同义
    - 'qwen/xxx' 等带前缀        -> 前缀即 provider，取前缀后为 model
    - 其他裸名                   -> 本地 transformers
    """
    s = spec.strip().replace("\\", "/")
    if not s:
        return "local", ""
    if os.path.isabs(s) or s.startswith("local/"):
        return "local", s if os.path.isabs(s) else s[len("local/"):]
    if s.startswith(("qwen3.8", "qwen3-8")):          # 裸模型名，如 qwen3.8-flash
        return "qwen38", s
    if "/" in s:
        p, n = s.split("/", 1)
        return ("qwen38" if p in ("qwen38", "qwen3.8") else p), n
    return "local", s


def _env_flag(name: str, default: bool) -> bool:
    """同名环境变量优先（1/true/yes/on），否则用 config 里的默认值。"""
    v = os.getenv(name, "").strip().lower()
    if v:
        return v in ("1", "true", "yes", "on")
    return bool(default)


def vlm_generator_model() -> str:
    """【不用改这里】读取下面的 VLM_GENERATOR_MODEL；命令行/环境变量可临时覆盖。"""
    return os.getenv("VLM_GENERATOR_MODEL") or VLM_GENERATOR_MODEL


def vlm_enable_thinking() -> bool:
    """【不用改这里】读取下面的 VLM_ENABLE_THINKING；命令行/环境变量可临时覆盖。"""
    return _env_flag("VLM_ENABLE_THINKING", VLM_ENABLE_THINKING)


def qwen_api_key() -> str:
    """DashScope 默认用哪个 key：优先 QWEN_API_KEY_2（旧 QWEN_API_KEY 可能欠费/限额）。"""
    return QWEN_API_KEY_2 or QWEN_API_KEY


# 读 .env（本仓库优先，再回退 backend/.env）
_load_dotenv(
    os.path.join(_REPO_DIR, ".env"),
    "/workspace/ai-camera-coach-app/backend/.env",
)

# ============================ 1. 7 维度定义 ============================
# 维度 key（固定顺序）
DIM_ORDER = ["ratio", "composition", "camera", "position", "pose", "focus", "color"]

# 维度中文名（生成阶段 prompt 组装用）
DIM_CN = {
    "ratio":       "Ratio 画面比例",
    "composition": "Composition 构图取景",
    "camera":      "Camera 机位视角",
    "position":    "Position 主体位置",
    "pose":        "Pose 姿态动作",
    "focus":       "Focus 对焦景深",
    "color":       "Color 色彩光影",
}

# AesRecon content 中与维度 key 一一对应的 tag 名
DIM_TAGS = {
    "ratio":       "Ratio",
    "composition": "Composition",
    "camera":      "Camera",
    "position":    "Position",
    "pose":        "Pose",
    "focus":       "Focus",
    "color":       "Color",
}

# 每个维度的条件提示（编码器输入，让模型只聚焦该维度视觉线索）
# ⚠️ 关键设计：每段 text 只提自己关注的维度，绝不出现其他维度名，
#    否则 7 段文本词面高度重合（都含全部维度词），编码出的 patch 向量区分度会很低。
DIM_CONDITION_PROMPTS = {
    "ratio": "Assess only the image aspect ratio and crop: is the frame tall or wide, how tight is the crop, how much margin or empty space surrounds the subject.",
    "composition": "Assess only framing and composition: rule of thirds, leading lines, foreground-background layering, and how elements are arranged within the frame.",
    "camera": "Assess only the camera position and shooting viewpoint: distance to subject, camera height, high or low angle, straight-on or tilted perspective.",
    "position": "Assess only where the subject is placed in the frame: centered or off-center, headroom, horizontal and vertical placement.",
    "pose": "Assess only the subject's pose and body action: posture, gestures, head and body direction, arrangement of arms and legs, and facial expression.",
    "focus": "Assess only focus and depth of field: what is sharp, background or foreground blur, bokeh, and focal plane control.",
    "color": "Assess only color and lighting: color temperature, saturation, brightness and exposure, contrast, highlights and shadows.",
}

# ============================ 2. 数据路径（共享盘绝对路径） ============================
# 大数据在共享盘（本机实际挂载点为 /datasets）；_data_path() 会自动回退到仓库内 dataset/ 软链。
_SHARED_DATA = os.getenv("DDGE_SHARED_DATA", "/gpfs3/area1/shared/ai-ddge")

# AesRecon 根目录（其下为 AesRecon_dataset/）
AESRECON_ROOT = os.getenv("AESRECON_ROOT", _data_path("AesRecon"))
# metadata.jsonl：每行 {image_path(参考好图), control_path(用户侧坏图), caption(带 7 维度 tag 的 content)}
AESRECON_METADATA = os.path.join(
    AESRECON_ROOT, "AesRecon_dataset/jsons/train/Stage2/metadata.jsonl"
)

# ============ 2.1 检索 / 参考语义 ============
# True  : 离线索引用「坏图」(control_path)，与用户照片匹配同类构图问题；
#         prompt 参考样图用「好图」(orig_image_path)（修好后目标效果）。
# False : 索引用「好图」(image_path)，prompt 参考样图同样用「好图」。
USE_POOR_AS_INDEX = True

# 入库按索引用图去重：AesRecon 中一张坏图会对应多张修好的好图（metadata 有 1365 张坏图被复用），
# 索引用坏图时若不去重，同一坏图会被多个样本各建一份 -> 检索分数相同、证据重复进 prompt。
DEDUP_INDEX_IMAGE = True

# ============================================================================
# 3. ★模型设置（最常改：① 编码器 1 行 / ② 生成 VLM 1 行 / ③ 思考模式开关）
# ============================================================================
# 本地模型根目录（相对名都拼到这里；换机器改这一行）
LOCAL_MODEL_ROOT = "/workspace/ai-camera-coach-app/backend/local_model_checkpoints"

# 运行设备：保持 "cuda"（跟随 CUDA_VISIBLE_DEVICES）。⚠️ 别写 "cuda:1" 这种带序号的，
# 过滤后只剩一张卡会越界；选卡请在 .env 里改 CUDA_VISIBLE_DEVICES。
DEVICE = "cuda"
DTYPE = "bfloat16"                 # 或 "float16"

# ---- ① 编码器（出 7 维特征；4B / 8B 任选）----
ENCODER_MODEL = "Qwen3-VL-4B-Instruct"
# ENCODER_MODEL = "Qwen3-VL-8B-Instruct"

# ---- ② 生成 VLM（改下面这一行即可切换；也可 CLI --vlm 或环境变量 VLM_GENERATOR_MODEL 覆盖）----
#   qwen3.8-flash                    DashScope 原生多模态接口（直接写模型名，支持思考）
#   qwen/qwen3-vl-flash-2026-01-22   DashScope OpenAI 兼容接口（默认）
#   vllm/Qwen3-VL-8B-Instruct        本地 vLLM（VLLM_BASE_URL）
#   local/Qwen3-VL-4B-Instruct       本地 transformers（权重名/绝对路径）
# VLM_GENERATOR_MODEL = os.getenv("VLM_GENERATOR_MODEL", "qwen/qwen3-vl-flash-2026-01-22")
VLM_GENERATOR_MODEL = "qwen3.8-flash"

# ---- ③ 思考模式：★ 就在这里改 ----
# True  = 开思考（更细，但慢很多：短请求实测 12.6s vs 1.1s）
# False = 关思考（快）
# 临时覆盖：命令行 --enable-thinking / 环境变量 VLM_ENABLE_THINKING=0
VLM_ENABLE_THINKING = 0

# ---- 服务地址 / Key（一般不用改）----
VLLM_BASE_URL = "http://localhost:8003/v1"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# qwen3.8-flash 走 DashScope 原生多模态接口（纯 requests，无需 SDK/代理），支持 enable_thinking
QWEN38_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_API_KEY_2 = os.getenv("QWEN_API_KEY_2", "")
QWEN_API_KEY_3 = os.getenv("QWEN_API_KEY_3", "")
# (key, 并发槽数)：第一个是有余额的 QWEN_API_KEY_2；旧 QWEN_API_KEY 若欠费，池子会自动换 key
QWEN_API_KEYS_CONCURRENCY = [
    (QWEN_API_KEY_2, 2),
    (QWEN_API_KEY_3, 2),
    (QWEN_API_KEY, 1),
]

# ---- 由上面的模型设置派生（不用改）----
ENCODER_MODEL_PATH = resolve_local_model(ENCODER_MODEL)
VLM_PROVIDER, VLM_MODEL_NAME = parse_model_spec(VLM_GENERATOR_MODEL)
# 使用的 GPU：单卡 "1" 或多卡 "0,1,2"；"auto"=自动选一张最空闲卡。多卡时批量推理自动多进程并行。
# ⚠️ 编号与 nvidia-smi 完全一致（下面已锁定 CUDA_DEVICE_ORDER=PCI_BUS_ID）。
#    本机 8 卡：nvidia-smi 0-3 = RTX PRO 5000 Blackwell(sm_120)，需 CUDA 12.8+ 的 torch；
#               nvidia-smi 4-7 = RTX 5880 Ada(sm_89)，torch cu126 可直接用。
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES") or "auto"   # 也可直接写 "6" 或 "4,5,6,7"
GPU_IDS = [int(x) for x in str(CUDA_VISIBLE_DEVICES).split(",") if x.strip().isdigit()] or None

# ---- 编码参数（一般不用改）----
IMAGE_MAX_PIXELS = 524_288         # 编码器输入图最大像素数（0.5MP=1024*512），控制 patch 数量
PATCH_POOL_K = 64                  # 每维 N 个 patch 池化成 K 个（K<=N）
ENCODE_MAX_CONCURRENCY = 5         # 逐维编码并发；1=串行最稳（多线程 GPU 前向易触发 BLAS 崩溃）
ENCODE_USE_BATCH = True            # 7 条件一次 batch 前向（最快，显存峰值最高）
# 共享视觉塔（方案 B）：视觉塔只前向 1 次，缓存特征平铺到 batch → 再跑一次 LLM 前向。
# 比 _batch_forward 约快 23%、显存更低（结果 cos≈0.994，纯 fp 数值差）。
# 索引库随此开关自动切换（True->B 库 / False->A 库，见第 4 节），向量与索引同源。
DDGE_SHARED_VISION = True          # 直接在这里改；环境变量 DDGE_SHARED_VISION=1|0 可覆盖
ENCODE_SHARED_VISION = _env_flag("DDGE_SHARED_VISION", DDGE_SHARED_VISION)

# ============================ 4. FAISS 索引（默认检索后端） ============================
# 方案：每维一个 faiss IndexFlatIP（暴力精确，无近似损失），faiss 内部 OpenMP 多线程，
#       7 维全部检索约 1.9s。离线重建时直接产 FAISS（见 get_7dimensions.py）。
# ⚠️ 每套向量方法各用一套索引库，互不覆盖：
#    A(现状 batch / ENCODE_SHARED_VISION=False) -> FAISS_INDEX_DIR_A（现有索引原样保留）
#    B(共享视觉塔 / ENCODE_SHARED_VISION=True)  -> FAISS_INDEX_DIR_B（新库，需用 B 重建）
# 生效目录 FAISS_INDEX_DIR / FAISS_SQ8_DIR 在 import 时按 ENCODE_SHARED_VISION 自动选中，
# 因此离线建库(get_7dimensions / main.py offline)、faiss-sq8、检索/服务(server/user) 全部自动
# 落到与当前向量方法一致的那套库（向量与索引同源）。
FAISS_INDEX_DIR_A = _data_path("faiss_index")          # A: 现状 batch 索引（勿删，保留备份）
FAISS_INDEX_DIR_B = _data_path("faiss_index_sv")       # B: 共享视觉塔索引
FAISS_INDEX_DIR = FAISS_INDEX_DIR_B if ENCODE_SHARED_VISION else FAISS_INDEX_DIR_A
# SQ8 量化目录（旧功能，不随 A/B 分库）：SQ8 无实际加速，不为 B 生成，勿额外产出。
# faiss-sq8 命令仍写到此旧目录（A 的旧 faiss_index_sq8 保留不动）。
FAISS_SQ8_DIR = _data_path("faiss_index_sq8")
# 当前生效方法标签（溯源 / 日志）
ENCODE_METHOD_LABEL = "B-shared-vision" if ENCODE_SHARED_VISION else "A-batch"
# faiss 内部 OpenMP 并行线程数（64 全核检索最快；服务 worker 为独立 spawn 进程，安全）
FAISS_NUM_THREADS = 64
# 离线重建时直接产出 FAISS 索引
BUILD_FAISS = True

# ============================ 5. 检索 / 生成 ============================
# ---- 粗召回：从向量库（2.16M 行 patch）里捞候选 ----
# 每个查询 patch 在 faiss 里召回最近 ANN_TOPK 个向量作为 MaxSim 候选池
# （61 个查询 patch × 50 ≈ 3050 个候选向量，再按 sample_id 聚合成候选样本）。
ANN_TOPK = 50

# ---- MaxSim 增强过滤（search_maxsim 输出前生效）----
# 增强 1：MaxSim 分数阈值，低于该阈值的样本直接剔除；None = 关闭
MAXSIM_SCORE_THRESHOLD = None
# 增强 2：patch 命中覆盖率（防止单 patch 刷分）
# N_q = 当前维度用户 patch 数量；样本至少被 min_coverage 比例的查询 patch 命中过才保留
MAXSIM_MIN_COVERAGE = 0.0

# ============ 5.1 方案生成（每维候选好图 -> 动态方案数） ============
# 方案数不再硬编码：方案 k = 每个维度取检索到的第 k 张候选好图 + 该维对应文本作为参考，生成 1 份方案。
#   实际方案数 = 各维候选好图数的最小值（动态）：某维只有 3 张好图就只出 3 份；有 8 张就出 8 份。
#   MIN_SOLUTIONS：检索时每维至少收集的候选好图张数（=方案数下限目标，默认 5）。
#   MAX_SOLUTIONS：方案数上限（None=不限，有多少张好图就出多少份方案）。
MIN_SOLUTIONS = 5
MAX_SOLUTIONS = 5
# 兼容别名：CLI `--n-solutions` / 服务参数的默认值；含义 = 方案数上限（None=不限）。
N_SOLUTIONS = MAX_SOLUTIONS
# 每份方案的最大生成 token 数（影响生成耗时：1536≈11s / 800≈5s）
GENERATE_MAX_NEW_TOKENS = 1536
# VLM 生成 API 请求对 429/5xx（限流/服务暂不可用）的最大重试次数（每次指数退避 2s,4s,8s...）
VLM_API_MAX_RETRIES = 3
# 每维最终候选样本数（MaxSim 聚合后保留的参考样本数，送入证据筛选）。
AGG_POOL_TOPK = 8
# 流式生成采样温度（>0 才有创造性差异）
GEN_TEMPERATURE = 0.9

# ============ 5.2 模型选择（见第 3 节 ★模型设置） ============
# 生成 VLM：VLM_GENERATOR_MODEL（qwen3.8-flash | qwen/... | vllm/... | local/...）
# 编码器：ENCODER_MODEL（4B / 8B）；图编模型见下方 IMAGE_EDIT_MODEL

# 图编模型：生成方案后并行对用户照片做实际编辑（qwen-image 系列，DashScope）
# 默认关闭：当前环境没有 qwen 预算时，先只做检索 + prompt 生成，避免意外调用付费接口。
# 触发方式：CLI `infer --image-edit` 强制开启；不传该参数时按本开关决定。
IMAGE_EDIT_ENABLED = True    # 默认是否图编；CLI `--image-edit` 可强制开启
IMAGE_EDIT_MODEL = "qwen/qwen-image-2.0-2026-03-03"
IMAGE_EDIT_SIZE = "1024*1024"
IMAGE_EDIT_PARALLEL = True   # 多方案并行编辑
IMAGE_EDIT_MAX_WORKERS = 2   # 图编并发数（DashScope qwen-image 限流严格，3 并发仍会 429，2 更稳）
# 批处理：图编提交后台异步执行，与下一张图的「检索+生成」重叠，缩短总墙钟时间；
# 批处理结束/进程退出前由 prompt.wait_pending_edits() 统一等待，结果不会丢。
# 可用环境变量 IMAGE_EDIT_ASYNC=0 关闭（关闭后图编同步阻塞、run_inference 返回已完成的编辑图）。
IMAGE_EDIT_ASYNC = os.getenv("IMAGE_EDIT_ASYNC", "1").strip().lower() in ("1", "true", "yes", "on")
# 图编 prompt 取哪段："full"=整份方案；"advice"=只取“# 针对性整改建议”段（更聚焦可执行）
IMAGE_EDIT_PROMPT_SECTION = "advice"

# ---- 并行与证据控制（初心版：每维独立检索最匹配坏图，展开 good 变体凑够候选好图）----
RETRIEVAL_PARALLEL = True       # 7 维检索并行
GENERATE_PARALLEL = False       # 本地 vllm/local 生成是否并行：本地 4B 建议串行生成，多方案并行长输出(1536)
                                # 在 24GB 卡上会 OOM；串行每次只占一份，峰值低、稳定。
                                # ⚠️ qwen(远程 API) 不读此开关：恒走下方多 key 并发池（默认 5 并发）。
INCLUDE_DIM_GOOD_IMAGE = True   # 该维证据是否把对应 good 图也送进 VLM（True/False 可自选）
# 每个维度最多展开多少个匹配坏图的 good 变体（兜底上限；实际按候选好图张数收集：
# 展开坏图变体直到凑够 MIN_SOLUTIONS 张候选好图即停）。73% 坏图只有 1 个 good 变体，
# 故默认 6 才能在多数命中下凑够 5 张好图。
VARIANTS_PER_DIM_POOR = 6
# 每个维度最多保留多少条候选好图（文本指导）。需 ≥ MIN_SOLUTIONS，否则方案数会被截断（默认 10）。
MAX_DIM_EVIDENCE = 10
# 候选好图是否“每张 poor 图只取它的第一张好图”（默认关=现状：把命中 poor 图的全部 good 变体铺开）。
# False：方案 k = 该维第 k 张候选好图，多份方案可能共用同一张 poor（只是换其不同 good 变体）；
# True ：方案 k = 第 k 张【不同】poor 图的第一张好图，方案间按 poor 图去重（方案数 ≤ 命中不同 poor 数）。
# 可用 DDGE_FIRST_GOOD_PER_POOR=1/0 覆盖（A/B 对比用）。
EVIDENCE_FIRST_GOOD_PER_POOR = _env_flag("DDGE_FIRST_GOOD_PER_POOR", True)
# 每个方案每个维度只送 1 张候选好图 + 对应 1 条文本（方案 k 用每维第 k 张；绝不发送该维全部文本）。
# 参考图数量因此固定为每维 1 张，无需再配置（原 GOOD_IMAGE_PER_DIM 已移除）。

# ============================ 6. 输出 ============================
OUTPUT_DIR = "./output/0910_sv_FIRST_GOOD_PER_POOR_qwen38flash1"
