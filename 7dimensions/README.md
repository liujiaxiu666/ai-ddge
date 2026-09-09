# 7 维度 DDGE 拍照指导系统

基于 **DDGE（Dimension-Decomposed Generative Embedding，维度分解生成式嵌入）** 的拍照指导系统。
对用户照片做 7 个独立维度的检索，生成多份可执行的拍照/后期整改方案，并可对照片实际编辑。

## 一、整体 Pipeline

```
1) 离线索引：AesRecon 样本 = 参考好图 + 坏图 + 带 7 维度 tag 的文本
            每张坏图用编码器跑 7 次（每次带一个维度的条件提示）→ 7 套 patch 多向量集合
            存入 FAISS（每维度一个索引，2.16M 行 / 2560 维）
2) 推理检索：用户照片同样 7 次条件编码 → 7 套查询 patch
            按维度对齐，每维独立做 ANN + MaxSim（集合-集合晚交互）检索
            命中「坏图」后展开其 good 变体，直到每维凑够至少 MIN_SOLUTIONS 张候选好图作为证据
3) 方案生成（可选）：方案数不再硬编码 —— 方案 k = 每个维度取第 k 张候选好图 + 对应文本
            （**每份方案每个维度只用 1 张候选好图 + 对应 1 条文本**）
            → VLM（qwen3-vl-flash）逐方案生成（方案数 = 各维候选好图数的最小值，动态；
            qwen 走多 key 并发池 5 并发，见配置）
            图编（qwen-image）可选并行编辑用户照片（CLI 用 --image-edit 触发）
```

7 个维度：`ratio`(画面比例) / `composition`(构图) / `camera`(机位视角) / `position`(主体位置) / `pose`(姿态动作) / `focus`(对焦景深) / `color`(色彩光影)。

**关键设计**：
- 每个维度的条件提示**只提自己的维度**，避免 7 段文本词面重合导致编码向量区分度低。
- 索引用**坏图**（匹配用户照片的同类构图问题），prompt 参考样图用**好图**（修好后的目标效果）。
- 每维**独立检索**（初心版）：7 个维度可以命中不同照片，各自展开证据，避免单张照片带偏所有维度。
- **方案数动态**（不再硬编码 5 个方案）：检索时每维至少凑够 `MIN_SOLUTIONS`(默认5) 张候选好图；
  方案 k = 每个维度取第 k 张候选好图 + 该维对应文本生成；实际方案数 = 各维候选好图数的最小值
  （某维只有 3 张好图就出 3 份；有 8 张就出 8 份），可用 `MAX_SOLUTIONS` 设上限。
- **每方案每维仅 1 图 + 1 文**：每份方案每个维度只使用第 k 张候选好图（1 张图 + 对应 1 条文本），
  绝不把该维检索到的全部文本一起送进 prompt。

## 二、目录结构与共享数据

```
7dimensions/                  # 本仓库
  ├─ config.py / main.py / server.py / encoder.py / faiss_index.py / user.py / prompt.py / image_edit.py ...
  ├─ dataset/            → 软链 → /gpfs3/area1/shared/ai-ddge     # 共享盘（只建这一个软链）
  │    ├─ faiss_index/        # 7 维 FAISS 检索索引
  │    ├─ faiss_index_sq8/    # SQ8 量化索引（备用）
  │    ├─ AesRecon/           # 数据集
  │    └─ data_batch/         # 华为给的小红书知识的批量数据
  ├─ output/                  # 推理结果输出
  └─ README.md
```

**只用建一个软链**：把共享盘根目录 `dataset/` 软链到本仓库，其下的 `faiss_index / AesRecon / ...` 全部自动可见：

```bash
# 任何一台实验室服务器上，进到仓库目录后执行（一个软链搞定）：
ln -s /gpfs3/area1/shared/ai-ddge 7dimensions/dataset
```

⚠️ **不同机器对 `/gpfs3/area1/shared` 的挂载点可能不同**（有的机器挂到 `/datasets`、有的直接可访问 `/gpfs3/area1/shared`）——软链目标请用你机器上对共享盘的**实际挂载路径**；NFS **服务器端规范路径固定是 `/gpfs3/area1/shared/ai-ddge/`**。换机器无需搬数据，重建这一个软链即可。

## 三、数据集（AesRecon）

- 根目录：`AESRECON_ROOT`（`./AesRecon`，软链到共享盘）
- 元数据：`AesRecon_dataset/jsons/train/Stage2/metadata.jsonl`，每行：
  ```
  {image_path(好图), control_path(坏图), caption(带 7 维度 tag 的 content)}
  ```
- content 含 7 个 tag：`<Ratio> <Composition> <Camera> <Position> <Pose> <Focus> <Color>`
- 规模：8168 条样本；**5061 张唯一坏图**（1365 张被多个好图复用，索引已去重）
- ⚠️ **73% 的坏图只有 1 个 good 变体** → 每维要凑够至少 `MIN_SOLUTIONS` 张候选好图，
  需要展开多个匹配坏图：检索阶段会持续展开坏图变体直到凑够（默认凑够 5 张即停）；
  `VARIANTS_PER_DIM_POOR` 只是最多展开坏图数的兜底上限。

## 四、模型 / 依赖

| 组件 | 默认 | 说明 |
|---|---|---|
| 编码器 | `Qwen3-VL-4B-Instruct`（本地权重）| 取 patch hidden-states 做多向量集合；8B/4B 可自选 |
| 生成 VLM | `qwen/qwen3-vl-flash-2026-01-22`（DashScope API）| `provider/model` 风格：`qwen/`(API) \| `vllm/`(本地 vLLM) \| `local/`(本地 transformers) |
| 图编 | `qwen/qwen-image-2.0-2026-03-03`（DashScope）| 方案生成后异步编辑照片 |
| 检索 | FAISS（IndexFlatIP × 7，暴力精确）| 每维约 308k 行 × 2560 维 |
| 环境 | conda `dcvc_rt` | `source /opt/conda/etc/profile.d/conda.sh && conda activate dcvc_rt` |

- `QWEN_API_KEY` 通过 `config._load_dotenv()` 从 `.env` **只读**加载（在 `config.py` 里可改 `.env` 路径，不修改该文件）。
- 本地模型权重目录：`config.LOCAL_MODEL_ROOT`（相对名都拼到这里）。
- **qwen(API) 多 key 并发池**：方案并行生成走 `config.QWEN_API_KEYS_CONCURRENCY`（默认
  `QWEN_API_KEY`×2、`QWEN_API_KEY_2`×2、`QWEN_API_KEY_3`×1 = 共 5 并发），用多个 key 分摊
  DashScope 单 key 限流（429）。**某个 key 失败（余额不足 / key 失效 / 429）时，任务自动
  换剩余 key 重试，不跳过该方案；只有所有 key 都失败才报错**。空 key 自动跳过（并发槽相应减少）。

## 五、配置（config.py 每个参数意义）

### 1. 数据路径
| 参数 | 含义 |
|---|---|
| `AESRECON_ROOT` / `AESRECON_METADATA` | 数据集根目录与 metadata.jsonl 路径 |
| `USE_POOR_AS_INDEX` | `True`=索引用坏图（默认）；`False`=用坏图对应的好图建索引 |
| `DEDUP_INDEX_IMAGE` | 同一张坏图只建一份索引（AesRecon 有 1365 张坏图被多好图复用，必须去重）|

### 2. 模型
| 参数 | 含义 |
|---|---|
| `DEVICE` / `DTYPE` | 运行设备 / 精度（`cuda` / `bfloat16`）|
| `LOCAL_MODEL_ROOT` | 本地模型根目录（相对名拼到这里）|
| `ENCODER_MODEL` | 编码器：`Qwen3-VL-4B-Instruct`（默认）/ `Qwen3-VL-8B-Instruct`，或任意目录名/绝对路径 |
| `VLM_GENERATOR_MODEL` | 生成 VLM，`provider/model` 风格：`qwen/...`(API) \| `vllm/...`(本地) \| `local/...`(transformers) |
| `QWEN_BASE_URL` / `QWEN_API_KEY` | DashScope API 地址与主密钥（key 从 .env 只读加载）|
| `QWEN_API_KEY_2` / `QWEN_API_KEY_3` | qwen 多 key 池的备用密钥（从 .env 只读；空 key 自动跳过）|
| `QWEN_API_KEYS_CONCURRENCY` | 多 key 并发槽位列表：默认 `[(key1,2),(key2,2),(key3,1)]` = 共 5 并发 |
| `VLLM_BASE_URL` | 本地 vLLM 地址（`vllm/` 提供方用）|
| `CUDA_VISIBLE_DEVICES` | `"auto"`=启动自动选最空闲卡；或指定卡号 |
| `IMAGE_MAX_PIXELS` | 编码器输入最大像素（0.5MP=524288），超限先缩放（影响 patch 数量上限，不影响是否池化）|
| `PATCH_POOL_K` | 每维 patch 集合**池化成 K 个（64）后入库**——FAISS 存的是池化后的 patch 向量，不是原始 patch hidden-states；只减少入库/检索行数，不减少模型前向时间 |

### 3. 检索
| 参数 | 含义 |
|---|---|
| `FAISS_INDEX_DIR` / `FAISS_SQ8_DIR` | faiss 索引目录 / SQ8 量化索引目录（备用）|
| `FAISS_NUM_THREADS` | faiss 并行线程数 |
| `BUILD_FAISS` | 离线重建时直接产出 FAISS 索引 |
| `ANN_TOPK` | 粗召回：每个查询 patch 召回最近多少个向量（61×50≈3050 候选）|
| `AGG_POOL_TOPK` | 每维 MaxSim 聚合后保留的候选**样本**数（进证据筛选前的中转池）|
| `MAXSIM_SCORE_THRESHOLD` | MaxSim 分数阈值过滤（`None`=关闭）|
| `MAXSIM_MIN_COVERAGE` | patch 命中覆盖率过滤（防单 patch 刷分，0.0=关闭）|

### 4. 证据筛选（决定最终给 VLM 什么）
| 参数 | 含义 |
|---|---|
| `VARIANTS_PER_DIM_POOR` | 每维最多展开多少个**不同坏图**的 good 变体（兜底上限，默认 6；实际按候选好图数收集，凑够 `MIN_SOLUTIONS` 张即停）|
| `MAX_DIM_EVIDENCE` | 每维最多保留多少条候选好图（需 ≥ `MIN_SOLUTIONS`，默认 10；`None`=不限）|
| `INCLUDE_DIM_GOOD_IMAGE` | 是否把该维 good 图也送进 VLM 做参考范例 |

### 5. 生成与并行
| 参数 | 含义 |
|---|---|
| `MIN_SOLUTIONS` | 检索时每维至少收集的候选好图张数（=方案数下限目标，默认 5）|
| `MAX_SOLUTIONS` | 方案数上限（`None`=不限，有多少张好图就出多少份方案）|
| `N_SOLUTIONS` | 兼容别名（=`MAX_SOLUTIONS`），同时是 CLI `--n-solutions` / 服务参数的默认值 |
| `GENERATE_MAX_NEW_TOKENS` | 每份方案最大 token 数（1536≈11s / 800≈5s）|
| `VLM_API_MAX_RETRIES` | VLM 生成 API 对 429/5xx 的最大重试次数（指数退避 2s,4s,8s...，默认 3）|
| `GEN_TEMPERATURE` | 采样温度（越大方案差异越大）|
| `RETRIEVAL_PARALLEL` | 7 维检索是否并行 |
| `GENERATE_PARALLEL` | 本地 vllm/local 各份方案是否并行；qwen(API) 恒走多 key 并发池（此开关不生效）|

### 6. 图编（异步）
| 参数 | 含义 |
|---|---|
| `IMAGE_EDIT_ENABLED` | 是否默认启用图编（CLI `infer --image-edit` 可强制开启）|
| `IMAGE_EDIT_MODEL` / `IMAGE_EDIT_SIZE` | 图编模型与输出尺寸 |
| `IMAGE_EDIT_PARALLEL` / `IMAGE_EDIT_MAX_WORKERS` | 是否并行 / 并发数（DashScope 限流，5 并发易 429）|
| `IMAGE_EDIT_PROMPT_SECTION` | 图编 prompt 取段：`"advice"`=只取整改建议段（默认）/ `"full"`=整份方案 |

### 7. 输出
| 参数 | 含义 |
|---|---|
| `OUTPUT_DIR` | 结果目录（`./output`）|

## 六、使用方法

```bash
cd 7dimensions
conda activate dcvc_rt   # 或 source /opt/conda/etc/profile.d/conda.sh && conda activate dcvc_rt
```

### 1. 离线重建 FAISS 索引（一次性，或换编码器后必须重跑）
```bash
# 单卡全量：约 4h（4B 编码 ~0.55 样本/s）
python main.py offline --start 0 --end 8168

# 多卡并行（每段一张卡）更快：各段写 faiss_index/{dim}.{part}.index
CUDA_VISIBLE_DEVICES=0 python main.py offline --start 0 --end 2042 --faiss-part 0 &
CUDA_VISIBLE_DEVICES=1 python main.py offline --start 2042 --end 4084 --faiss-part 1 &
# ... 最后合并
python main.py faiss-merge --parts 4
```
> ⚠️ 换编码器（4B↔8B）后向量维度变了（2560↔4096），**必须重建**整个索引。

### 2. 推理（三种模式：单张 / 文件夹，均输出 JSON 汇总 + 各模块计时）
```bash
# 模式 1：只检索（不做 VLM 生成 / 图编）→ 输出 evidence 与检索计时
python main.py retrieve --input /path/to/photo.jpg --output ./output

# 模式 2：检索 + 生成若干份 prompt（每份=每维第 k 张候选好图+文本，方案数动态；默认不带图编）
python main.py infer --input /path/to/photo.jpg --output ./output --n-solutions 5

# 模式 3：检索 + 生成 prompt + 图像编辑（追加 --image-edit）
python main.py infer --input /path/to/photo.jpg --output ./output --n-solutions 5 --image-edit
```

输出到 `output/<图片名>/`：

| 模式 | 产出 | JSON 汇总（含计时） |
|---|---|---|
| `retrieve` | 仅检索证据 | `retrieval_summary.json`：evidence_by_dim + timing_s{retrieval, total} |
| `infer` | 检索 + 动态份数方案 | `solutions.json`：evidence_by_dim + solutions + timing_s{retrieval, vlm_generation, image_edit, total} |
| `infer --image-edit` | 检索 + 动态份数方案 + 对应编辑图 | `solutions.json`（同上 + edit_images + 图编计时） |

> `--n-solutions N` 是**方案数上限**（不传/默认=不限）：实际方案数 = 各维候选好图数的最小值
> （检索时每维至少凑 `MIN_SOLUTIONS` 张候选好图，通常默认出 ≥5 份方案）。
> 文件夹模式默认最多处理前 8 张（`--max-images 0` 或单独 `--max-images` = 不限）。
> 图编也可用环境变量 `IMAGE_EDIT_ENABLED=true` 默认开启（CLI `--image-edit` 优先）。
> `timing_s` 已拆分：`encode`（7 维 VLM 编码）+ `faiss_search`（FAISS 检索）+ `vlm_generation` + `image_edit`；
> `retrieval` = `encode` + `faiss_search`（保留兼容）。波动大时先看慢在编码还是检索。
> **失败也必有 JSON**：即使个别/全部方案生成失败，仍会写 `solutions.json`——成功方案正常输出，
> 失败方案带 `"error"` 字段，另有 `errors` 汇总和完整 `timing_s`（检索/VLM生成/图编/总计），便于看时延。

### 3. 常驻服务（实时：模型+索引常驻内存，图编异步）
```bash
# 指定一张空闲 GPU（auto 可能选到被占满的卡）
CUDA_VISIBLE_DEVICES=0 nohup python main.py serve --port 8100 --n-solutions 5 > /tmp/server.log 2>&1 &

# 调用
curl -X POST localhost:8100/infer -H 'Content-Type: application/json' \
     -d '{"image_base64": "<base64 of jpg>", "stem": "photo1"}'
curl localhost:8100/health
```
返回：`{"solutions":[动态份数方案文本], "evidence_by_dim", "retrieval_s", "edit_status":"processing"}`，图编在后台完成后写入输出目录。

### 4. 其它命令
```bash
python main.py faiss-merge --parts 4   # 合并并行重建的分段索引
python main.py faiss-sq8               # fp32 索引转 SQ8（弃用备用，检索慢 10 倍）
```

## 七、延时实测与可能原因

| 阶段 | 实测 | 说明 |
|---|---|---|
| 7 维编码（4B）| ~2.75s | batch 前向（7 条件一次跑完）|
| 7 维 FAISS 检索 | ~2s | 全库 2.16M 行暴力精确 |
| **只检索模式（CLI）** | ~5-10s | 编码 + 检索即结束，不跑 VLM（`main.py retrieve`）|
| 索引加载 | 冷 ~26s / 热 ~4.5s | 索引读入内存；常驻服务只付一次 |
| VLM 生成 5 方案 | ~11s | 5 路并行 qwen-flash（1536 tokens；方案数=各维候选好图数，默认至少 5）|
| 图编 5 张 | ~30s+ | API 远程 + DashScope 限流（3 并发）|
| **CLI 单张（1 方案）** | ~13s | 检索 8.5 + VLM 4.8（400 tokens）|
| **服务单张（默认≥5 方案）** | ~30s | 检索 + VLM 11（受节点负载影响会波动）|

### 10 秒目标为什么难达到
1. **编码 ~2.75s + 检索 ~2s** 是刚性成本（常驻服务下约 5s）。
2. **VLM 默认 5 方案 ~11s**（1536 tokens）是主要瓶颈 → 压到 800 tokens 约 5s。
3. **图编 5 张物理上不可能进 10s**（单张 API 就要 10s+ 且限流），故服务设计为**图编异步**（10s 内先返回方案文本）。
4. 在共享计算节点上，检索耗时可能因 GPU/CPU 竞争而波动，独占节点更稳定。

### 已知坑（均已解决/规避）
- 服务进程内直接跑 faiss+torch 会 OpenBLAS 崩溃 → 用独立 spawn worker 进程隔离。
- PyTorch 禁止 fork 子进程初始化 CUDA → worker 必须 `spawn`。
- 图编 5 并发触发 DashScope 429 → 限 3 并发 + 退避重试。
- 某 qwen key 余额不足/失效（401/403）→ 多 key 并发池自动换剩余 key 重试该方案（不跳过）；
  即使**全部 key 都失败**，也只把该方案记为 `error` 并继续，`solutions.json` 仍会落盘（含计时），
  不会让整张图的结果丢失。
- 生成阶段 qwen 被限流（429 `limit_requests`）→ `_api_stream` 先**指数退避重试**（2s/4s/8s...，
  次数 `VLM_API_MAX_RETRIES`）；同一 key 重试耗尽再由并发池换 key 重试；仍失败则该方案记 error 继续。
- 后台 `> log` 需 `PYTHONUNBUFFERED=1` 才实时落盘。
