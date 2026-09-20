# -*- coding: utf-8 -*-
"""
CLI 入口：统一调度「离线入库」与「推理生成（多方案 + 图编）」。

==================== 0) 环境准备（一次） ====================
  source /opt/conda/etc/profile.d/conda.sh && conda activate dcvc_rt
  cd /workspace/ai-ddge/7dimensions
  # API Key 自动从 /workspace/ai-camera-coach-app/backend/.env 读取（QWEN_API_KEY）

==================== 1) 离线入库（建向量库，可选） ====================
  # 重建前 100 个样本的索引（坏图建索引 + 去重 + 0.5MP）
  python main.py offline --start 0 --end 100 --rebuild
  # 全量（8168 样本；0.5MP 下约 40-60 分钟）
  python main.py offline --start 0 --end 8168 --rebuild
  # 只处理前 N 条（调试）
  python main.py offline --limit 50

==================== 2) 推理（三种模式，均输出 JSON 汇总 + 各模块计时） ====================
  # 模式 1：只检索（不生成 prompt / 不图编）-> output/<图片名>/retrieval_summary.json
  python main.py retrieve --input /path/to/user_photo.jpg
  # 模式 2：检索 + 生成若干份方案（每份=每维第 k 张候选好图+文本，方案数动态）-> output/<图片名>/solutions.json
  python main.py infer --input /path/to/user_photo.jpg --n-solutions 5
  python main.py infer --input /workspace/ai-camera-coach-app/backend/test_data/0715/dataset_link/jingxuan --max-images 0
  # 模式 3：检索 + 生成方案 + 图像编辑（追加 --image-edit；缺省时按 config.IMAGE_EDIT_ENABLED）
  python main.py infer --input /workspace/ai-camera-coach-app/backend/test_data/0715/flower100_20 --max-images 0  --n-solutions 5 --image-edit 
  # 整个文件夹（默认最多处理前 8 张，超出会被截断；--max-images 0 或单独 --max-images = 不限）
  python main.py infer --input /path/to/img_folder --max-images
  # 自定义：方案数上限 / 输出目录 / 生成 token 上限
  python main.py infer --input xxx.jpg --n-solutions 3 --output ./output --max-new-tokens 1536

==================== 3) 先入库再推理 ====================
  python main.py all --input xxx.jpg --end 100

==================== 模型选择（config.py 或环境变量） ====================
  # 生成 VLM：qwen/...（DashScope API，默认）| vllm/...（本地 vLLM）| local/...（本地 transformers）
  VLM_GENERATOR_MODEL=qwen/qwen3-vl-flash-2026-01-22  python main.py infer --input xxx.jpg
  VLM_GENERATOR_MODEL=vllm/Qwen3-VL-8B-Instruct        python main.py infer --input xxx.jpg
  VLM_GENERATOR_MODEL=local/Qwen3-VL-8B-Instruct       python main.py infer --input xxx.jpg
  # 编码器 4B/8B（config.ENCODER_MODEL，改完要重建向量库）：
  #   ENCODER_MODEL=Qwen3-VL-4B-Instruct / Qwen3-VL-8B-Instruct

  # 起 vLLM 服务（VLM_GENERATOR_MODEL=vllm/... 前先跑；端口/模型与 config.VLLM_BASE_URL 一致）
  #   pip install vllm
  #   vllm serve /workspace/ai-camera-coach-app/backend/local_model_checkpoints/Qwen3-VL-8B-Instruct \
  #     --port 8003 --max-model-len 16384 --gpu-memory-utilization 0.9

  # 图编开关与模型（可选）
  #   CLI 加 --image-edit 即触发图编；也可用环境变量 IMAGE_EDIT_ENABLED=true 默认开启
  IMAGE_EDIT_MODEL=qwen/qwen-image-2.0-2026-03-03 python main.py infer --input xxx.jpg --image-edit
  IMAGE_EDIT_ENABLED=true python main.py infer --input xxx.jpg   # 无需 --image-edit 也图编
  # 证据/并行开关（config.py 里改也可）
  #   INCLUDE_DIM_GOOD_IMAGE=true/false  该维指导是否带 good 图进 prompt
  #   RETRIEVAL_PARALLEL / GENERATE_PARALLEL  并行开关
  #   MIN_SOLUTIONS / MAX_SOLUTIONS  每维候选好图目标张数 / 方案数上限
  #   VARIANTS_PER_DIM_POOR / MAX_DIM_EVIDENCE  坏图展开兜底 / 每维候选好图保留上限

输出：
  模式 1 retrieve           -> output/<图片名>/retrieval_summary.json（evidence + 检索计时）
  模式 2 infer              -> output/<图片名>/solutions.json（evidence + N 方案 + timing_s）
  模式 3 infer --image-edit -> 同模式 2，另有 solution_1..N.png（含图编计时）
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# OpenBLAS "too many memory regions" 段错误修复：必须在任何 BLAS/numpy/torch 加载前锁定线程数
# （连续多图推理时多次 "Program is Terminated" 崩溃）。encoder.py 也有兜底。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "8"))
os.environ.setdefault("MKL_NUM_THREADS", "1")

from config import (FAISS_INDEX_DIR, FAISS_SQ8_DIR, GENERATE_MAX_NEW_TOKENS,  # noqa: E402
                    GPU_IDS, N_SOLUTIONS, OUTPUT_DIR)
# 注意：不要在顶层 import get_7dimensions/user（会连带初始化 encoder/CUDA）。
# 需要时在函数内 import，这样多进程并行时各 worker 能各自指定 GPU。


def _apply_vlm_env(vlm: str | None, enable_thinking: bool | None) -> None:
    """把 --vlm / --enable-thinking 写入环境变量，供 config.vlm_generator_model() 动态读取
    （也保证 spawn 子进程能继承）。vlm 示例：qwen3.8-flash（DashScope 原生接口，可思考）。"""
    if vlm:
        os.environ["VLM_GENERATOR_MODEL"] = vlm
    if enable_thinking is not None:
        os.environ["VLM_ENABLE_THINKING"] = "1" if enable_thinking else "0"
    if vlm or enable_thinking is not None:
        print(f"[main] 生成VLM: {os.environ.get('VLM_GENERATOR_MODEL','<config默认>')} | "
              f"enable_thinking={os.environ.get('VLM_ENABLE_THINKING','<auto>')}", flush=True)


def _add_offline_args(p):
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 条样本")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--rebuild", action="store_true", help="先清空旧集合再重建")
    p.add_argument("--faiss-part", type=int, default=None,
                   help="并行重建分段号(0..N-1)，各段写 faiss_index/{dim}.{part}.*，最后用 faiss-merge 合并")


def _add_infer_args(p):
    p.add_argument("--input", required=True, help="用户图片路径 或 图片文件夹路径")
    p.add_argument("--output", type=str, default=OUTPUT_DIR, help="结果保存目录")
    p.add_argument("--max-new-tokens", type=int, default=GENERATE_MAX_NEW_TOKENS,
                   help="生成模型最大新 token 数")
    p.add_argument("--n-solutions", type=int, default=N_SOLUTIONS,
                   help="方案数上限（默认不限；实际方案数=各维候选好图数的最小值，检索时每维至少凑 MIN_SOLUTIONS 张）")
    p.add_argument("--max-images", type=int, default=8, nargs="?", const=0,
                   help="文件夹模式下最多处理多少张图（0 或不给数值 = 不限）")
    p.add_argument("--image-edit", action="store_true", default=None,
                   help="执行图像编辑（仅对生成命令生效；缺省时遵循 config.IMAGE_EDIT_ENABLED）")
    p.add_argument("--vlm", default=None,
                   help="生成VLM（provider/model 或直接写模型名），如 qwen3.8-flash；缺省用 config.VLM_GENERATOR_MODEL")
    p.add_argument("--enable-thinking", action="store_true", default=None,
                   help="开启思考模式（qwen38 缺省自动开启；此处可强制开启）")


def main() -> None:
    parser = argparse.ArgumentParser(description="AesRecon 7 维度 DDGE 拍照指导系统")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_off = sub.add_parser("offline", help="离线入库")
    _add_offline_args(p_off)

    p_mg = sub.add_parser("faiss-merge", help="合并分段重建的 faiss 索引")
    p_mg.add_argument("--out", default=FAISS_INDEX_DIR)
    p_mg.add_argument("--parts", type=int, default=4, help="分段数")

    p_sq = sub.add_parser("faiss-sq8", help="fp32 faiss 索引转 SQ8 量化版（保留原索引，输出到 FAISS_SQ8_DIR）")
    p_sq.add_argument("--src", default=FAISS_INDEX_DIR)
    p_sq.add_argument("--out", default=FAISS_SQ8_DIR)

    p_srv = sub.add_parser("serve", help="启动常驻 HTTP 推理服务（模型+索引常驻，图编异步）")
    p_srv.add_argument("--port", type=int, default=8100)
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--n-solutions", type=int, default=N_SOLUTIONS)
    p_srv.add_argument("--max-new-tokens", type=int, default=GENERATE_MAX_NEW_TOKENS)
    p_srv.add_argument("--output", type=str, default=OUTPUT_DIR)
    p_srv.add_argument("--vlm", default=None, help="生成VLM，如 qwen3.8-flash")
    p_srv.add_argument("--enable-thinking", action="store_true", default=None, help="开启思考模式")

    p_ret = sub.add_parser("retrieve", help="只做检索并输出 evidence 汇总 JSON")
    _add_infer_args(p_ret)

    p_inf = sub.add_parser("infer", help="检索 + 生成 prompt 拍照指导")
    _add_infer_args(p_inf)

    p_all = sub.add_parser("all", help="先离线入库再推理")
    _add_offline_args(p_all)
    p_all.add_argument("--input", required=True, help="用户图片路径 或 图片文件夹路径")
    p_all.add_argument("--output", type=str, default=OUTPUT_DIR)
    p_all.add_argument("--max-new-tokens", type=int, default=GENERATE_MAX_NEW_TOKENS)
    p_all.add_argument("--n-solutions", type=int, default=N_SOLUTIONS)
    p_all.add_argument("--max-images", type=int, default=8, nargs="?", const=0,
                       help="文件夹模式最多处理多少张（0 或不给数值 = 不限）")
    p_all.add_argument("--vlm", default=None, help="生成VLM，如 qwen3.8-flash")
    p_all.add_argument("--enable-thinking", action="store_true", default=None, help="开启思考模式")

    args = parser.parse_args()

    if args.cmd == "faiss-merge":
        from faiss_index import FaissIndex
        FaissIndex.merge_parts(args.out, n_parts=args.parts)
        return

    if args.cmd == "faiss-sq8":
        from faiss_index import FaissIndex
        FaissIndex.convert_to_sq8(args.src, args.out)
        return

    if args.cmd == "serve":
        _apply_vlm_env(getattr(args, "vlm", None), getattr(args, "enable_thinking", None))
        from server import run_server
        run_server(port=args.port, host=args.host, n_solutions=args.n_solutions,
                   max_new_tokens=args.max_new_tokens, output=args.output)
        return

    if args.cmd == "offline":
        from get_7dimensions import offline_build
        offline_build(limit=args.limit, start=args.start, end=args.end, rebuild=args.rebuild,
                      faiss_part=args.faiss_part)
        return

    if args.cmd == "all":
        from get_7dimensions import offline_build
        print("[main] 先执行离线入库 ...")
        offline_build(limit=args.limit, start=args.start, end=args.end, rebuild=args.rebuild,
                      faiss_part=args.faiss_part)
        _do_infer(args.input, args.output, args.max_new_tokens, args.n_solutions, args.max_images,
                  vlm=args.vlm, enable_thinking=args.enable_thinking)
        return

    if args.cmd == "retrieve":
        _do_infer(args.input, args.output, args.max_new_tokens, args.n_solutions, args.max_images,
                  generate_prompt=False, image_edit=None,
                  vlm=args.vlm, enable_thinking=args.enable_thinking)
        return

    if args.cmd == "infer":
        _do_infer(args.input, args.output, args.max_new_tokens, args.n_solutions, args.max_images,
                  generate_prompt=True, image_edit=args.image_edit,
                  vlm=args.vlm, enable_thinking=args.enable_thinking)


def _infer_worker(gpu: int, paths: list, output: str, max_new_tokens: int, n_solutions: int,
                 generate_prompt: bool = True, image_edit: bool | None = None,
                 vlm: str | None = None) -> None:
    """多进程并行 worker：绑定一张卡，处理分配给它的图片子集（spawn 子进程）。
    image_edit=None 时由 config.IMAGE_EDIT_ENABLED 决定是否图编。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if vlm:
        os.environ["VLM_GENERATOR_MODEL"] = vlm
    from user import run_inference
    for i, p in enumerate(paths):
        print(f"\n[main:{gpu}] 处理 {i + 1}/{len(paths)}: {p}", flush=True)
        try:
            _e, _s, js_path, edit_paths, _r = run_inference(
                p, output_dir=output, max_new_tokens=max_new_tokens, n_solutions=n_solutions,
                generate_prompt=generate_prompt, image_edit=image_edit, vlm_model=vlm,
            )
            print(f"[main:{gpu}] 已保存: {js_path}（{len(edit_paths)} 张编辑图）", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[main:{gpu}] 处理失败 {p}: {exc}", flush=True)

    # 批处理结束：等待后台异步图编全部完成，避免最后几张的编辑结果随进程退出丢失
    from prompt import wait_pending_edits
    print(f"[main:{gpu}] 全部图片处理完，等待后台图编完成 ...", flush=True)
    wait_pending_edits()


def _infer_parallel(paths: list, output: str, max_new_tokens: int, n_solutions: int, gpus: list,
                   generate_prompt: bool = True, image_edit: bool | None = None,
                   vlm: str | None = None) -> None:
    """多卡并行：把图片均分给 len(gpus) 个 spawn 进程，每进程绑定一张卡。"""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = []
    for i, gpu in enumerate(gpus):
        chunk = paths[i::len(gpus)]
        if not chunk:
            continue
        p = ctx.Process(target=_infer_worker, daemon=False,
                        args=(gpu, chunk, output, max_new_tokens, n_solutions,
                              generate_prompt, image_edit, vlm))
        p.start()
        procs.append(p)
        print(f"[main] worker 启动: 卡{gpu} 处理 {len(chunk)} 张", flush=True)
    for p in procs:
        p.join()


def _do_infer(input_path: str, output: str, max_new_tokens: int, n_solutions: int, max_images: int,
              generate_prompt: bool = True, image_edit: bool | None = None,
              vlm: str | None = None, enable_thinking: bool | None = None) -> None:
    _apply_vlm_env(vlm, enable_thinking)
    from user import resolve_input_paths
    paths = resolve_input_paths(input_path)
    if max_images and len(paths) > max_images:
        paths = paths[:max_images]
        print(f"[main] 文件夹模式，最多处理 {max_images} 张：{paths}")

    if GPU_IDS and len(GPU_IDS) > 1 and len(paths) > 1:
        print(f"[main] 使用 {len(GPU_IDS)} 张卡并行: {GPU_IDS}")
        _infer_parallel(paths, output, max_new_tokens, n_solutions, GPU_IDS,
                        generate_prompt=generate_prompt, image_edit=image_edit, vlm=vlm)
        return

    from user import run_inference
    for i, p in enumerate(paths):
        print(f"\n[main] 处理 {i + 1}/{len(paths)}: {p}")
        evidence, solutions, js_path, edit_paths, retr_s = run_inference(
            p, output_dir=output, max_new_tokens=max_new_tokens, n_solutions=n_solutions,
            generate_prompt=generate_prompt, image_edit=image_edit, vlm_model=vlm,
        )
        print(f"[main] 已保存: {js_path}（{len(edit_paths)} 张编辑图）")

    # 批处理结束：等待后台异步图编完成（异步模式下 edit_paths 为空属正常，结果写入 solutions.json）
    from prompt import wait_pending_edits
    print("[main] 全部图片处理完，等待后台图编完成 ...", flush=True)
    wait_pending_edits()


if __name__ == "__main__":
    main()
