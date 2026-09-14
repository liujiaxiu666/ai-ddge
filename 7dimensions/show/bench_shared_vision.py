# -*- coding: utf-8 -*-
"""
7 维度 patch 特征提取：三种生产前向路径的 耗时/显存 对比 + 结果一致性回归。

  A) _batch_forward                 —— 现状默认（batch=7，视觉塔内部仍算 7 遍相同图）
  B) _batch_forward_shared_vision   —— 共享视觉塔（视觉塔 1 次 + 一次 batch LLM；
                                        config.ENCODE_SHARED_VISION=True 时生产走此路径）
  C) get_dim_condition_patch_embedding 循环 —— 逐维 7 次全前向

预期一致性（最终池化+归一化 patch 集余弦）：
  B vs C ≈ 0.9995+；B vs A 与 C vs A 都 ≈ 0.994（现状 batch-vs-single 的固有数值差）。

用法：cd 7dimensions && /opt/conda/envs/dcvc_rt/bin/python bench_shared_vision.py
     [--gpu 0..3(必须 Ada, torch2.7 不兼容 Blackwell)] [--img 图片路径] [--repeat N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="1")
parser.add_argument("--img", default=None)
parser.add_argument("--repeat", type=int, default=4)
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "8"))
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from config import DIM_CONDITION_PROMPTS, DIM_ORDER  # noqa: E402
from encoder import DdgeEncoder, _limit_pixels  # noqa: E402


def cos_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() * b.float()).sum(-1).mean().item()


def run_timed(fn, repeat: int):
    """运行 repeat 次，返回 (耗时列表, 输出, 峰值显存GB)。每次前重置峰值统计。"""
    times, peak = [], 0.0
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peak = max(peak, torch.cuda.max_memory_allocated() / 1024**3)
    return times, out, peak


def main() -> None:
    print(f"[bench] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
          f"torch={torch.__version__} gpu={torch.cuda.get_device_name()}")
    enc = DdgeEncoder()
    conds = {d: DIM_CONDITION_PROMPTS[d] for d in DIM_ORDER}

    img_path = args.img
    if not img_path:
        cand = "/datasets/ai-ddge/AesRecon/AesRecon_dataset/images/poor_images/3160_poor.jpg"
        img_path = cand if os.path.exists(cand) else None
    pil_img = _limit_pixels(Image.open(img_path).convert("RGB")) if img_path and os.path.exists(img_path) \
        else Image.new("RGB", (1024, 512), (128, 90, 60))
    print(f"[bench] 测试图: {img_path or '(合成)'} 尺寸={pil_img.size}")

    # warmup（kernel 编译 / 首次分配 / rope_deltas 初始化）
    enc._batch_forward_shared_vision(pil_img, conds)
    torch.cuda.synchronize()

    tC, outC, peakC = run_timed(
        lambda: {d: enc.get_dim_condition_patch_embedding(pil_img, conds[d]) for d in DIM_ORDER}, args.repeat)
    tA, outA, peakA = run_timed(lambda: enc._batch_forward(pil_img, conds), args.repeat)
    tB, outB, peakB = run_timed(lambda: enc._batch_forward_shared_vision(pil_img, conds), args.repeat)

    print("\n========== 耗时(ms/张, 7维度) ==========")
    for name, t in [("C 逐维7次全前向", tC), ("A batch(视觉算7遍)", tA), ("B 共享视觉塔", tB)]:
        ts = sorted(t)
        print(f"{name:22s} min={ts[0]*1000:7.1f}  mean={sum(t)/len(t)*1000:7.1f}  (n={len(t)})")

    print("\n========== 峰值显存 allocated (GB, 含模型~8G) ==========")
    for name, p in [("C 逐维7次全前向", peakC), ("A batch", peakA), ("B 共享视觉塔", peakB)]:
        print(f"{name:22s} {p:6.2f} GB")

    print("\n========== 一致性 (最终池化+归一化 patch 集余弦) ==========")
    print(f"  {'dim':12s} {'B vs A':>10s} {'B vs C':>10s} {'C vs A':>10s}")
    for d in DIM_ORDER:
        print(f"  {d:12s} {cos_mean(outB[d], outA[d]):10.6f} {cos_mean(outB[d], outC[d]):10.6f} "
              f"{cos_mean(outC[d], outA[d]):10.6f}")


if __name__ == "__main__":
    main()
