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


# ---- 加载 .env（本仓库 .env 优先，再回退 backend/.env 补 QWEN_API_KEY），失败静默 ----
def _load_dotenv(*paths: str) -> None:
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


_load_dotenv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),  # 本仓库 .env（GPU / key 可自配）
    "/workspace/ai-camera-coach-app/backend/.env",                       # 回退：老项目 QWEN_API_KEY
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
# 所有大数据在共享盘 `/gpfs3/area1/shared/ai-ddge/`（NFS 规范路径）。不同机器对该共享点的
# 挂载路径可能不同（例如本机挂到 /datasets），故 _data_path() 优先用规范绝对路径，若本机
# 未挂载则回退到仓库内 dataset/ 软链（dataset -> 共享盘根）。也可用环境变量覆盖。
_SHARED_DATA = os.getenv("DDGE_SHARED_DATA", "/gpfs3/area1/shared/ai-ddge")
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _data_path(rel: str) -> str:
    """共享盘规范绝对路径优先；本机挂载点不同（不可达）时回退仓库内 dataset/ 软链。
    ⚠️ 子目录尚不存在时也返回本地 dataset 挂载点路径（调用方会 os.makedirs），
       避免落到本机并不存在的 /gpfs3 规范路径上。"""
    abs_p = os.path.join(_SHARED_DATA, rel)
    if os.path.exists(abs_p):
        return abs_p
    link_p = os.path.join(_REPO_DIR, "dataset", rel)
    if os.path.isdir(os.path.dirname(link_p)):      # 本地 dataset 挂载点在，优先走它
        return link_p
    return abs_p


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

# ============ 3. 模型配置（编码 / 生成VLM，可自选） ============
# 运行设备：保持 "cuda"（跟随 CUDA_VISIBLE_DEVICES 指定的卡，见下）。⚠️ 不要写成 "cuda:1"
# 这类带序号的写法——CUDA_VISIBLE_DEVICES 过滤后只剩一张卡（映射为 cuda:0），写死了会越界报
# invalid device ordinal。选哪张卡请在 .env 里改 CUDA_VISIBLE_DEVICES（0/1/2/3 或 auto）。
DEVICE = "cuda"
DTYPE = "bfloat16"                # 或 "float16"

# 本地模型根目录（相对名都拼到这里；换机器时改这一行为你的权重目录即可）
LOCAL_MODEL_ROOT = "/workspace/ai-camera-coach-app/backend/local_model_checkpoints"

# ---- 编码器（4B / 8B 任选，或写任意本地目录名/绝对路径）----
ENCODER_MODEL = "Qwen3-VL-4B-Instruct"      # 默认已切 4B
# ENCODER_MODEL = "Qwen3-VL-8B-Instruct"

# ---- 生成 VLM（provider/model 风格，同 backend/.env）----
VLM_GENERATOR_MODEL = "qwen/qwen3-vl-flash-2026-01-22"        # 远程 DashScope
# VLM_GENERATOR_MODEL = "vllm/Qwen3-VL-8B-Instruct"            # 本地 vLLM
# VLM_GENERATOR_MODEL = "local/Qwen3-VL-8B-Instruct"           # 本地 transformers
# VLM_GENERATOR_MODEL = "/workspace/ai-camera-coach-app/backend/local_model_checkpoints/Qwen2-VL-2B-Instruct"
# 与编码器同一模型（Qwen3-VL-4B）：多模型缓存只驻留一份权重，显存最省、质量好、实测不 OOM
# VLM_GENERATOR_MODEL = "local/Qwen3-VL-4B-Instruct"
# VLM_GENERATOR_MODEL = os.getenv("VLM_GENERATOR_MODEL", "local/Qwen3-VL-8B-Instruct")

# ---- 服务地址 / Key ----
VLLM_BASE_URL = "http://localhost:8003/v1"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")

# ---- qwen 多 key 并发池（DashScope 限流/余额不足规避）----
# 单 key 并发过高易 429，且个别 key 可能余额不足：用多个 key 分摊，默认 3 个 key ——
# 前两个各 2 并发、第三个 1 并发（共 5 并发）。某个 key 失败时并发池自动换剩余 key 重试（不跳过）。
# 从 .env 读取 QWEN_API_KEY / QWEN_API_KEY_2 / QWEN_API_KEY_3（空 key 自动跳过，并发槽相应减少）。
QWEN_API_KEY_2 = os.getenv("QWEN_API_KEY_2", "")
QWEN_API_KEY_3 = os.getenv("QWEN_API_KEY_3", "")
# (key, 并发槽数) 列表，顺序即权重轮询顺序
QWEN_API_KEYS_CONCURRENCY = [
    (QWEN_API_KEY, 2),
    (QWEN_API_KEY_2, 2),
    (QWEN_API_KEY_3, 1),
]


# ---- 解析助手 ----
def resolve_local_model(spec: str) -> str:
    """把 'local/xxx'、相对名、或绝对路径解析为绝对权重路径。"""
    s = spec.strip().replace("\\", "/")
    s = os.path.expanduser(s)
    if s.startswith("local/"):
        s = s[len("local/"):]
    if os.path.isabs(s):
        return s
    return os.path.join(LOCAL_MODEL_ROOT, s)


def parse_model_spec(spec: str):
    """返回 (provider, model_name)；绝对路径 / local/ 前缀都视为本地模型。"""
    s = spec.strip().replace("\\", "/")
    if not s:
        return "local", ""
    if os.path.isabs(s) or s.startswith("local/"):
        return "local", s if os.path.isabs(s) else s[len("local/"):]
    if "/" in s:
        p, n = s.split("/", 1)
        return p, n
    return "local", s


# 编码器 / 生成 VLM 的最终解析结果
ENCODER_MODEL_PATH = resolve_local_model(ENCODER_MODEL)
VLM_PROVIDER, VLM_MODEL_NAME = parse_model_spec(VLM_GENERATOR_MODEL)

# 使用的 GPU：单卡数字（"1"）或多卡逗号列表（"0,1,2"）；"auto"=自动选一张最空闲卡。
# 在本仓库 .env 里写 CUDA_VISIBLE_DEVICES=... 即可。多卡时文件夹批量推理自动多进程并行（每卡一个 worker）。
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "auto")
# 解析出的卡列表（多卡或单卡；"auto"/空 -> None）
GPU_IDS = [int(x) for x in str(CUDA_VISIBLE_DEVICES).split(",") if x.strip().isdigit()] or None

# 编码器输入图像最大像素数（控制视觉 patch 数量）
IMAGE_MAX_PIXELS = 524_288         # 0.5MP = 1024*512

# patch 集合池化：每维度 N 个 patch 池化成 K 个（K<=N）
PATCH_POOL_K = 64

# 编码并发：逐维编码时同时编码的维度数。
# 1 = 串行（最稳：单线程 GPU 前向不触发 OpenBLAS 段错误，显存峰值 ≈ 模型+1 图，连续多张稳定，
#     速度与 batch 相当，约 2.8s/张）；>1 并发虽快但多线程 GPU 前向易触发 BLAS 崩溃（默认 1）。
ENCODE_MAX_CONCURRENCY = 5
# 是否用 batch 前向（7 个条件一次跑完，最快但显存峰值最高，单卡 24GB 连续处理多张易 OOM）。
# 默认 False：用串行逐维编码，显存水位低、连续多张稳定。
# ENCODE_USE_BATCH = False
ENCODE_USE_BATCH = True

# 共享视觉塔编码（方案 B）：视觉塔（Qwen3-VL DeepStack，输出与文本完全无关）只跑 1 次，
# 缓存 pooler_output + deepstack_features 平铺到 batch 行，再做一次 batch LLM 前向。
#   - 相对 _batch_forward（视觉算 7 遍）：约快 ~23%、显存更低；
#   - 结果与「逐维单图前向」cos≈0.999、与 _batch_forward cos≈0.994（均为 FA/batch-vs-single
#     的 fp 数值差，非语义差异；B 自身 run-to-run 逐位确定）；
# ⚠️ 需直连 HF 内部子模块（lm.language_model / get_image_features / compute_3d_position_ids）、
#     transformers 大版本升级可能失效。
# 索引库也随此开关自动切换（True->B 库，False->A 库，见第 4 节）：向量与索引同源，杜绝混搭。
# 简单切换：直接改下面默认值 1/0，或环境变量 DDGE_SHARED_VISION=1/0 覆盖。
ENCODE_SHARED_VISION = os.getenv("DDGE_SHARED_VISION", "1").strip().lower() in ("1", "true", "yes", "on")

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

# ============ 5.2 模型选择（见第 3 节统一定义） ============
# 生成 VLM：VLM_GENERATOR_MODEL（qwen/... | vllm/... | local/...）
# 编码器：ENCODER_MODEL（4B / 8B）；图编模型见下方 IMAGE_EDIT_MODEL

# 图编模型：生成方案后并行对用户照片做实际编辑（qwen-image 系列，DashScope）
# 默认关闭：当前环境没有 qwen 预算时，先只做检索 + prompt 生成，避免意外调用付费接口。
# 触发方式：CLI `infer --image-edit` 强制开启；不传该参数时按本开关决定。
IMAGE_EDIT_ENABLED = os.getenv("IMAGE_EDIT_ENABLED", "true").lower() in ("1", "true", "yes", "on")
IMAGE_EDIT_MODEL = "qwen/qwen-image-2.0-2026-03-03"
IMAGE_EDIT_SIZE = "1024*1024"
IMAGE_EDIT_PARALLEL = True   # 多方案并行编辑
IMAGE_EDIT_MAX_WORKERS = 2   # 图编并发数（DashScope qwen-image 限流严格，3 并发仍会 429，2 更稳）
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
EVIDENCE_FIRST_GOOD_PER_POOR = os.getenv("DDGE_FIRST_GOOD_PER_POOR", "1").strip().lower() in ("1", "true", "yes", "on")
# 每个方案每个维度只送 1 张候选好图 + 对应 1 条文本（方案 k 用每维第 k 张；绝不发送该维全部文本）。
# 参考图数量因此固定为每维 1 张，无需再配置（原 GOOD_IMAGE_PER_DIM 已移除）。

# ============================ 6. 输出 ============================
OUTPUT_DIR = "./output/0907_sv_FIRST_GOOD_PER_POOR"
