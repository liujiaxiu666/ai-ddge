# -*- coding: utf-8 -*-
"""
将 dimension_stats.csv 中的各维度评估结果绘制为柱状图进行比较。

数据形态：每一行是一个 (dim, kind) 组合
  dim  : ratio / composition / camera / position / pose / focus / color
  kind : analysis（原始分析） 与 advice（改善建议）
  指标 : acc_strict（严格准确率） acc_weighted（加权准确率）
         以及 correct/partial/wrong 三档对错数量（total = 50）

生成图片：
  1) dimension_acc_bars.png
     分组柱状图：x 轴为 7 个维度，柱按 kind 区分（analysis vs advice），
     分别绘制 acc_strict 与 acc_weighted 两个子图，柱顶标注数值。
  2) dimension_distribution_bars.png
     堆叠柱状图：每个 (dim, kind) 显示 correct / partial / wrong 的数量构成。
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")  # 无显示环境，直接存图
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                         "dimension_stats.csv")
# OUT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = "/workspace/ai-ddge/7dimensions/output/0909_no_rag/qwen_eval/dimension_stats.csv"
OUT_DIR = "/workspace/ai-ddge/7dimensions/output/0909_no_rag/qwen_eval"


def _pick_label(csv_path: str) -> str:
    """由 CSV 路径推断数据集标签（用于标题）。
    取路径中 output 之后的一段，剥掉 *_qwen_eval/_gpt_eval/_eval 后缀：
      .../output/0909_no_rag/qwen_eval/csv  -> 0909_no_rag
      .../output/0907_sv_qwen_eval/csv      -> 0907_sv
    路径里没有 output 时退化为取最后一段再剥后缀。"""
    parts = os.path.normpath(os.path.dirname(os.path.abspath(csv_path))).split(os.sep)
    comp = parts[-1]
    if "output" in parts:
        idx = parts.index("output")
        if idx + 1 < len(parts):
            comp = parts[idx + 1]
    for suf in ("_qwen_eval", "_gpt_eval", "_eval"):
        if comp.endswith(suf) and len(comp) > len(suf):
            return comp[:-len(suf)]
    return comp

# 维度显示顺序与名称（沿用 CSV 中的英文，避免缺中文字体导致乱码）
KINDS = ["analysis", "advice"]
KIND_LABEL = {"analysis": "analysis", "advice": "advice"}
DIM_ORDER = None  # None 表示按 CSV 出现的顺序

DPI = 150


def load_data():
    rows = []
    with open(CSV_PATH, encoding="utf-8-sig") as f:  # utf-8-sig 自动去掉 BOM
        for r in csv.DictReader(f):
            rows.append(r)
    dims = list(dict.fromkeys(r["dim"] for r in rows))  # 保留 CSV 顺序且去重
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    # 每格样本量 N：优先用 CSV 的 total 列，缺失时用 correct+partial+wrong 求和
    first = rows[0]
    if first.get("total"):
        n_total = int(first["total"])
    else:
        n_total = sum(int(first.get(c, 0)) for c in ("correct", "partial", "wrong"))
    return rows, dims, kinds, n_total


def pick(rows, dim, kind):
    return next(r for r in rows if r["dim"] == dim and r["kind"] == kind)


def draw_acc_bars(rows, dims, kinds, n_total, label):
    """图 1：acc_strict 与 acc_weighted 的 analysis vs advice 分组柱状图。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
    colors = {"analysis": "#4C72B0", "advice": "#DD8452"}
    width = 0.36

    for ax, metric in zip(axes, ["acc_strict", "acc_weighted"]):
        x = range(len(dims))
        for i, kind in enumerate(kinds):
            vals = [float(pick(rows, d, kind)[metric]) for d in dims]
            offset = (i - 0.5) * (width + 0.05)
            bars = ax.bar([xi + offset for xi in x], vals, width,
                          label=KIND_LABEL[kind], color=colors[kind],
                          edgecolor="black", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(list(x))
        ax.set_xticklabels(dims, fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xlabel("dimension")
        ax.set_title(f"{label} · {metric}", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(frameon=False, fontsize=10)

    fig.suptitle(f"[{label}] Per-dimension accuracy: analysis vs advice (n={n_total})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT_DIR, "dimension_acc_bars.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)


def draw_distribution_bars(rows, dims, kinds, n_total, label):
    """图 2：correct / partial / wrong 数量堆叠柱状图，按 kind 分两个并排子图。"""
    # 颜色语义：correct=绿, partial=橙, wrong=红
    cats = [("correct", "#2CA02C"), ("partial", "#FFC000"), ("wrong", "#D62728")]
    fig, axes = plt.subplots(1, len(kinds), figsize=(15, 6), sharey=True)
    for ax, kind in zip(axes, kinds):
        x = list(range(len(dims)))
        bottom = [0] * len(dims)
        for cat, color in cats:
            vals = [int(pick(rows, d, kind)[cat]) for d in dims]
            bars = ax.bar(x, vals, 0.6, bottom=bottom, color=color,
                          edgecolor="white", linewidth=0.6, label=cat)
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0:
                    ax.text(xi, b + v / 2, str(v), ha="center", va="center",
                            fontsize=10, color="white", fontweight="bold")
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(x)
        ax.set_xticklabels(dims, fontsize=11)
        ax.set_ylim(0, n_total + 5)
        ax.set_xlabel("dimension")
        ax.set_title(f"[{label}] kind = {kind}", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(frameon=False, fontsize=10, loc="upper right")
        if ax is axes[0]:
            ax.set_ylabel(f"count (total = {n_total})")

    fig.suptitle(f"[{label}] correct / partial / wrong distribution (n={n_total} per cell)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT_DIR, "dimension_distribution_bars.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)


def main():
    global CSV_PATH, OUT_DIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH, help="dimension_stats.csv 路径")
    ap.add_argument("--out", default=None, help="输出图片目录（默认与 CSV 同目录）")
    ap.add_argument("--label", default=None,
                    help="图表标题里的数据集标签（默认按 CSV 路径推断，如 0909_no_rag）")
    args = ap.parse_args()
    CSV_PATH = args.csv
    OUT_DIR = args.out or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(OUT_DIR, exist_ok=True)
    label = args.label or _pick_label(args.csv)

    rows, dims, kinds, n_total = load_data()
    print(f"label={label} | n_total={n_total} | dims: {dims} | kinds: {kinds}")
    draw_acc_bars(rows, dims, kinds, n_total, label)
    draw_distribution_bars(rows, dims, kinds, n_total, label)
    print("done.")


if __name__ == "__main__":
    main()
