"""
0830 批次各阶段时延统计（基于 output/0830 每个图片文件夹 solutions.json 的 timing_s）
- 阶段：retrieval（检索）、vlm_generation（VLM 生成）、image_edit（图像编辑）、total（总耗时）
- 输出每个阶段的 均值/中位数/最小/最大/标准差 及样本数，并生成柱状图与箱线图
- 说明：该批次 image_edit 全为 0（未启用图编），故 total = retrieval + vlm_generation
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 中文字体支持
for font_path in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                  '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
                  '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc']:
    if Path(font_path).exists():
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

# ==================== 配置 ====================
RESULTS_DIR = Path('/workspace/ai-ddge/7dimensions/output/0901')  # 测试结果目录
OUTPUT_DIR = RESULTS_DIR                                           # 图表保存到结果目录下
STAGES = ['retrieval', 'vlm_generation', 'image_edit', 'total']
STAGE_CN = {
    'retrieval': '检索 Retrieval',
    'vlm_generation': 'VLM 生成 VLM Generation',
    'image_edit': '图像编辑 Image Edit',
    'total': '总耗时 Total',
}
STAGE_EN = {
    'retrieval': 'Retrieval',
    'vlm_generation': 'VLM Generation',
    'image_edit': 'Image Edit',
    'total': 'Total',
}

# ==================== 数据收集 ====================
def collect_cases(results_dir):
    """
    扫描结果目录下所有图片文件夹的 solutions.json，提取 timing_s 各阶段时延。
    要求 retrieval 与 vlm_generation 都存在；image_edit/total 可为缺失（按 None 处理）。
    """
    cases = []
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        js = folder / 'solutions.json'
        if not js.exists():
            continue
        with open(js, 'r', encoding='utf-8') as f:
            data = json.load(f)

        timings = data.get('timing_s', {})
        case_info = {
            'case': folder.name,
            'retrieval': timings.get('retrieval'),
            'vlm_generation': timings.get('vlm_generation'),
            'image_edit': timings.get('image_edit'),
            'total': timings.get('total'),
        }
        # 必须有检索与 VLM 阶段
        if case_info['retrieval'] is None or case_info['vlm_generation'] is None:
            print(f"⚠️ 跳过 {folder.name}，缺少 retrieval/vlm_generation")
            continue
        cases.append(case_info)

    return cases

# ==================== 计算统计量 ====================
def compute_stats(cases, stage):
    """对某阶段的所有样本计算 均值/中位数/最小/最大/标准差。"""
    vals = [c[stage] for c in cases if c.get(stage) is not None]
    if not vals:
        return {'n': 0}
    arr = np.array(vals, dtype=float)
    return {
        'n': len(arr),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'std': float(arr.std()),
    }

# ==================== 主函数 ====================
def main():
    cases = collect_cases(RESULTS_DIR)
    if not cases:
        print("❌ 没有有效的 case 数据，退出。")
        return
    print(f"✅ 共读取 {len(cases)} 个有效 case（每个来自一个图片文件夹的 solutions.json）")

    # ---------- 打印统计表 ----------
    print("\n" + "=" * 78)
    print(f"{'阶段':<30}{'N':>4}{'均值':>11}{'中位数':>11}{'最小':>9}{'最大':>9}{'标准差':>11}")
    print("-" * 78)
    stats = {}
    for stage in STAGES:
        st = compute_stats(cases, stage)
        stats[stage] = st
        if st['n'] == 0:
            print(f"{STAGE_CN[stage]:<30}{'0':>4}{'-':>11}{'-':>11}{'-':>9}{'-':>9}{'-':>11}")
            continue
        print(f"{STAGE_CN[stage]:<30}{st['n']:>4}{st['mean']:>9.2f}s{st['median']:>9.2f}s"
              f"{st['min']:>7.2f}s{st['max']:>7.2f}s{st['std']:>9.2f}s")
    print("=" * 78)
    print("说明：该批次 image_edit 全为 0（未启用图编），因此 total ≈ retrieval + vlm_generation")

    # total 自洽性校验（允许 0.01s 的 JSON 两位小数舍入误差）
    checked = mismatch = 0
    for c in cases:
        if c['total'] is not None:
            s = c['retrieval'] + c['vlm_generation'] + (c['image_edit'] or 0)
            checked += 1
            if abs(s - c['total']) > 0.011:
                mismatch += 1
    print(f"   total 自洽性校验：{checked} 个 case 中 {mismatch} 个不一致（容差 0.01s，其余为舍入误差）")

    # ---------- 柱状图（Mean ± Std + Min-Max 范围线），英文标签 ----------
    labels = [STAGE_EN[s] for s in STAGES]
    means = [stats[s]['mean'] if stats[s]['n'] else 0 for s in STAGES]
    stds = [stats[s]['std'] if stats[s]['n'] else 0 for s in STAGES]
    mins = [stats[s]['min'] if stats[s]['n'] else 0 for s in STAGES]
    maxs = [stats[s]['max'] if stats[s]['n'] else 0 for s in STAGES]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, means, width=0.55, color='#1f77b4', edgecolor='black',
                  yerr=stds, capsize=4, label='Mean ± Std')
    ax.errorbar(x, (np.array(mins) + np.array(maxs)) / 2,
                yerr=(np.array(maxs) - np.array(mins)) / 2,
                fmt='none', ecolor='#d62728', capsize=6, label='Min-Max range')

    ax.set_ylabel('Latency (s)')
    ax.set_title(f'Stage Timing Stats - 0830 batch (N={len(cases)})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5, f'{v:.1f}s',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    out_bar = OUTPUT_DIR / '0830_stage_timing_stats.png'
    plt.savefig(out_bar, dpi=150)
    plt.close()
    print(f"\n✅ 各阶段时延统计图已保存: {out_bar}")

    # ---------- 各阶段分布箱线图（英文标签） ----------
    data_list = [[c[s] for c in cases if c.get(s) is not None] for s in STAGES]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_list, patch_artist=True, boxprops=dict(facecolor='#a6c8e8'))
    ax = plt.gca()
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    plt.ylabel('Latency (s)')
    plt.title(f'Stage Timing Distribution - 0830 batch (N={len(cases)})')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_box = OUTPUT_DIR / '0830_stage_timing_boxplot.png'
    plt.savefig(out_box, dpi=150)
    plt.close()
    print(f"✅ 各阶段时延箱线图已保存: {out_box}")

    print("\n🎯 所有结果已生成。")

if __name__ == '__main__':
    main()