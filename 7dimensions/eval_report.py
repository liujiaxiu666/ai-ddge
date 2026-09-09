#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 qwen/gpt 评估结果（output/<xxx>_eval/raw/<图片>/verdicts.json）渲染成报告：
  - 左侧：用户原图
  - 右侧：按【方案(solution) × 维度(dim)】用表格展示 analysis / advice 的
          label(correct/wrong/partial/null) + reason，label 按颜色着色。
支持输出 PDF 与 PPTX（默认都出；可 --skip-pdf / --skip-pptx）。

用法：
  python eval_report.py --verdicts-dir output/0907_sv_qwen_eval/raw --out output/0907_sv_qwen_eval
  python eval_report.py --verdicts-dir .../raw --only 0249a71 --skip-pptx   # 只出 1 张的 pdf
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from PIL import Image

DIM_ORDER = ["ratio", "composition", "camera", "position", "pose", "focus", "color"]
DIM_CN = {
    "ratio": "1 画面比例", "composition": "2 构图取景", "camera": "3 机位视角",
    "position": "4 主体位置", "pose": "5 姿态动作", "focus": "6 对焦景深", "color": "7 色彩光影",
}

# label -> 显示色
LABEL_COLOR = {
    "correct": "#1a7f37",
    "partial": "#b76e00",
    "wrong": "#cf222e",
    "null": "#8b949e",
}
LABEL_CN = {"correct": "正确", "partial": "部分正确", "wrong": "错误", "null": "未提供"}

_DEFAULT_VERDICTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "output", "0907_sv_qwen_eval", "raw")


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
def collect_verdicts(verdicts_dir: str, only: str | None = None) -> list[dict]:
    out = []
    for d in sorted(glob.glob(os.path.join(verdicts_dir, "*", ""))):
        if not os.path.isfile(os.path.join(d, "verdicts.json")):
            continue
        if only and only not in os.path.basename(os.path.normpath(d)):
            continue
        with open(os.path.join(d, "verdicts.json"), "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def group_by_solution(verdicts: list[dict]) -> dict:
    """{solution: {dim: {"analysis": {label,reason}, "advice": {...}}}} 按 DIM_ORDER 排序。"""
    sols: dict[int, dict] = {}
    for it in verdicts:
        sol = it.get("solution")
        dim = it.get("dim")
        if dim not in DIM_ORDER:
            continue
        sols.setdefault(sol, {})[dim] = {
            "analysis": it.get("analysis") or {},
            "advice": it.get("advice") or {},
        }
    for s in sols.values():
        s = dict(sorted(s.items(), key=lambda kv: DIM_ORDER.index(kv[0])))
    return dict(sorted(sols.items()))


def img_fit(path: str, box_w: float, box_h: float):
    if not path or not os.path.exists(path):
        return box_w * 0.6, box_h * 0.6
    with Image.open(path) as im:
        iw, ih = im.size
    r = min(box_w / iw, box_h / ih)
    return iw * r, ih * r


def _image_stream(path: str):
    """读原图为 JPEG BytesIO（兼容 mpo 等 python-pptx 不支持的格式；JPEG 体积小），返回 (stream, w, h)。"""
    import io
    im = Image.open(path).convert("RGB")
    w, h = im.size
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    return buf, w, h


def parse_solution_dims_orig(text: str) -> dict:
    """从方案正文解析 每维 {analysis, advice} 原文（复用 evaluate_qwen 的解析）。"""
    try:
        from evaluate_qwen import parse_solution_dims
        return parse_solution_dims(text)
    except Exception:  # noqa: BLE001
        return {}


def load_original_solutions(results_dir: str, image_name: str) -> dict:
    """{solution: {dim: {analysis, advice}}} 原文；找不到返回空。"""
    p = os.path.join(results_dir, image_name, "solutions.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for s in d.get("solutions", []):
        if not isinstance(s, dict) or s.get("index") is None:
            continue
        out[s["index"]] = parse_solution_dims_orig(s.get("text", ""))
    return out


def default_results_dir(verdicts_dir: str) -> str:
    """由 raw 目录反推原 results 目录：.../<xxx>_eval/raw -> .../<xxx>"""
    parent = os.path.dirname(os.path.normpath(verdicts_dir))
    base = os.path.basename(parent)
    for suf in ("_qwen_eval", "_gpt_eval", "_eval"):
        if base.endswith(suf):
            cand = os.path.join(os.path.dirname(parent), base[:-len(suf)])
            if os.path.isdir(cand):
                return cand
    return parent


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def render_pdf(items, out_path: str, page_size=(13.333, 7.5)):
    from reportlab.lib.pagesizes import inch
    from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                    Table, TableStyle, Spacer,
                                    NextPageTemplate, PageBreak)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch as _in
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.lib.utils import ImageReader
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    FONT = "STSong-Light"

    PW, PH = page_size[0] * inch, page_size[1] * inch
    LEFT_W = 4.3 * inch
    GAP = 0.3 * inch
    RIGHT_W = PW - LEFT_W - GAP - 0.5 * inch

    st_head = ParagraphStyle("head", fontName=FONT, fontSize=13, leading=16,
                             textColor=colors.HexColor("#1f3864"))
    st_sol = ParagraphStyle("sol", fontName=FONT, fontSize=11, leading=14,
                            textColor=colors.HexColor("#1f3864"), spaceBefore=8)
    st_cell = ParagraphStyle("cell", fontName=FONT, fontSize=7.2, leading=9.5,
                             wordWrap="CJK")
    st_orig = ParagraphStyle("orig", fontName=FONT, fontSize=6.8, leading=8.8,
                             wordWrap="CJK", textColor=colors.HexColor("#24292f"))
    st_ver = ParagraphStyle("ver", fontName=FONT, fontSize=6.8, leading=8.8,
                            wordWrap="CJK")
    st_dim = ParagraphStyle("dim", fontName=FONT, fontSize=8.2, leading=10,
                            wordWrap="CJK")

    class Doc(BaseDocTemplate):
        pass

    # 每张图一个 PageTemplate：onPage 闭包绑定该图自己的原图，避免“共用一张图”
    def make_draw_left(img_path: str, name: str):
        def _draw(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(colors.HexColor("#f6f8fa"))
            canvas.rect(0, 0, LEFT_W + GAP, PH, stroke=0, fill=1)
            canvas.setFillColor(colors.HexColor("#1f3864"))
            canvas.setFont(FONT, 11)
            canvas.drawString(0.25 * inch, PH - 0.45 * inch, f"原图：{name}")
            if img_path and os.path.exists(img_path):
                bw, bh = LEFT_W + GAP - 0.5 * inch, PH - 1.4 * inch
                iw, ih = img_fit(img_path, bw, bh)
                ix = 0.25 * inch + (bw - iw) / 2
                iy = PH - 1.0 * inch - ih
                canvas.drawImage(ImageReader(img_path), ix, iy, iw, ih, mask="auto")
            canvas.setFont(FONT, 7.5)
            canvas.setFillColor(colors.HexColor("#57606a"))
            canvas.drawString(0.25 * inch, 0.25 * inch,
                              "左：原图  |  右：qwen 评估（原·缺陷分析/建议 + label + reason）")
            canvas.restoreState()
        return _draw

    doc = Doc(out_path, pagesize=(PW, PH),
              leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0)
    frame = Frame(LEFT_W + GAP, 0.4 * inch, RIGHT_W, PH - 0.8 * inch, id="f")
    templates = [PageTemplate(id=f"p{i}", frames=[frame],
                              onPage=make_draw_left(it.get("input_image", ""),
                                                    it.get("image", "")))
                 for i, it in enumerate(items)]
    doc.addPageTemplates(templates)

    story = []
    for i, it in enumerate(items):
        if i > 0:                                   # 切到该图自己的模板，并强制分页
            story.append(NextPageTemplate(f"p{i}"))
            story.append(PageBreak())
        name = it.get("image", "")
        sols = group_by_solution(it.get("dims", []))
        orig_sols = it.get("_orig", {}) or {}
        story.append(Paragraph(f"评估报告 · {name}", st_head))
        for sol, dims in sols.items():
            story.append(Paragraph(f"方案 {sol}  （{len(dims)} 个维度）", st_sol))
            od = orig_sols.get(sol, {})
            W = RIGHT_W
            cw = [0.62 * inch,
                  (W - 0.62 * inch) * 0.30, (W - 0.62 * inch) * 0.20,
                  (W - 0.62 * inch) * 0.30, (W - 0.62 * inch) * 0.20]
            rows = [[Paragraph("维度", st_dim), Paragraph("原·缺陷分析", st_dim),
                     Paragraph("分析判定", st_dim),
                     Paragraph("原·整改建议", st_dim),
                     Paragraph("建议判定", st_dim)]]
            for dim in DIM_ORDER:
                if dim not in dims:
                    continue
                e = dims[dim]
                o = od.get(dim) or {}
                orig_an = (o.get("analysis") or "").strip() or "（未提供）"
                orig_ad = (o.get("advice") or "").strip() or "（未提供）"
                rows.append([
                    Paragraph(DIM_CN[dim], st_dim),
                    Paragraph(orig_an, st_orig),
                    _label_reason_para(e["analysis"], st_ver),
                    Paragraph(orig_ad, st_orig),
                    _label_reason_para(e["advice"], st_ver),
                ])
            t = Table(rows, colWidths=cw, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f7")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d7de")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ]))
            story.append(t)
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))
    doc.build(story)
    print(f"[report] PDF: {out_path}")


def _label_reason_para(e: dict, style) -> Paragraph:
    """analysis/advice -> Paragraph: 着色 label + reason 换行。"""
    from reportlab.platypus import Paragraph
    label = e.get("label") or "null"
    label = label if label in LABEL_CN else "null"
    color = LABEL_COLOR.get(label, "#8b949e")
    reason = (e.get("reason") or "").strip()
    if not reason:
        reason = "（无理由）"
    return Paragraph(
        f'<font color="{color}"><b>[{LABEL_CN[label]}]</b></font> {reason}',
        style)


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------
def render_pptx(items, out_path: str):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    RGB = lambda h: RGBColor(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    for it in items:
        name = it.get("image", "")
        img_path = it.get("input_image", "")
        sols = group_by_solution(it.get("dims", []))
        orig_sols = it.get("_orig", {}) or {}
        for sol, dims in sols.items():
            slide = prs.slides.add_slide(blank)
            # 左侧原图（统一转 PNG 字节流，兼容 .mpo 等非常规格式）
            if img_path and os.path.exists(img_path):
                stream, iw, ih = _image_stream(img_path)
                r = min(3.9 / iw, 6.4 / ih)
                w, h = Inches(iw * r), Inches(ih * r)
                slide.shapes.add_picture(stream, Inches(0.25), Inches(0.6),
                                         width=w, height=h)
            tb = slide.shapes.add_textbox(Inches(0.25), Inches(0.1), Inches(4.0), Inches(0.4))
            tb.text_frame.text = f"{name}  方案 {sol}"
            tb.text_frame.paragraphs[0].font.size = Pt(14)
            tb.text_frame.paragraphs[0].font.bold = True
            # 右侧表格：维度 | 原·缺陷分析 | 分析判定 | 原·整改建议 | 建议判定
            od = orig_sols.get(sol, {})
            nrows = len(dims) + 1
            gt = slide.shapes.add_table(nrows, 5, Inches(4.5), Inches(0.35),
                                        Inches(8.55), Inches(6.9)).table
            for c, htxt in enumerate(["维度", "原·缺陷分析", "分析判定",
                                      "原·整改建议", "建议判定"]):
                gt.cell(0, c).text = htxt
            for i, dim in enumerate(DIM_ORDER):
                if dim not in dims:
                    continue
                e = dims[dim]
                o = od.get(dim) or {}
                gt.cell(i + 1, 0).text = DIM_CN[dim]
                orig_an = (o.get("analysis") or "").strip() or "（未提供）"
                orig_ad = (o.get("advice") or "").strip() or "（未提供）"
                gt.cell(i + 1, 1).text = orig_an
                gt.cell(i + 1, 3).text = orig_ad
                for c, key in ((2, "analysis"), (4, "advice")):
                    ev = e[key]
                    label = (ev.get("label") or "null")
                    label = label if label in LABEL_CN else "null"
                    reason = (ev.get("reason") or "").strip() or "（无理由）"
                    gt.cell(i + 1, c).text = f"[{LABEL_CN[label]}] {reason}"
            # 样式
            for row in gt.rows:
                row.height = Inches(6.9 / (len(dims) + 1))
            for ri in range(nrows):
                for ci in range(3):
                    cell = gt.cell(ri, ci)
                    for para in cell.text_frame.paragraphs:
                        para.font.size = Pt(9 if ci == 0 else 8)
                        if ci == 0:
                            para.alignment = PP_ALIGN.CENTER
                            para.font.bold = True
                    if ri == 0:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = RGB("#eef1f7")
            # 给 label 上色（简单处理：第2/3列首个 [正确/部分正确/错误] 着色）
            _color_labels(gt)
    prs.save(out_path)
    print(f"[report] PPTX: {out_path}")


def _color_labels(table):
    """只给判定列(2=分析判定,4=建议判定)的 [正确/部分正确/错误/未提供] 前缀着色。"""
    from pptx.dml.color import RGBColor
    for ri in range(1, len(table.rows)):
        for ci in (2, 4):
            cell = table.cell(ri, ci)
            for para in cell.text_frame.paragraphs:
                for run in para.runs:
                    for lab, cn in LABEL_CN.items():
                        if f"[{cn}]" in run.text:
                            run.font.color.rgb = RGBColor(*_hex(LABEL_COLOR[lab]))
                            break


def _hex(h):
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="qwen/gpt 评估结果 -> PDF/PPTX 报告")
    ap.add_argument("--verdicts-dir", default=_DEFAULT_VERDICTS,
                    help="raw 目录（含各 <图片>/verdicts.json）")
    ap.add_argument("--solutions-dir", default=None,
                    help="原 results 目录（含各 <图片>/solutions.json，用于取原文对照；"
                         "默认从 verdicts-dir 反推，如 .../0907_sv_qwen_eval/raw -> .../0907_sv）")
    ap.add_argument("--out", default=None, help="输出目录（默认=<verdicts-dir> 的上级）")
    ap.add_argument("--only", default=None, help="只处理含该关键字的图片")
    ap.add_argument("--limit", type=int, default=0, help="最多前 N 张（0=全部）")
    ap.add_argument("--skip-pdf", action="store_true")
    ap.add_argument("--skip-pptx", action="store_true")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    items = collect_verdicts(args.verdicts_dir, only=args.only)
    if args.limit > 0:
        items = items[:args.limit]
    if not items:
        raise SystemExit(f"[report] {args.verdicts_dir} 下没有 verdicts.json")
    results_dir = args.solutions_dir or default_results_dir(args.verdicts_dir)
    n_orig = 0
    for it in items:
        it["_orig"] = load_original_solutions(results_dir, it.get("image", ""))
        if it["_orig"]:
            n_orig += 1
    if not n_orig:
        print(f"[report] 警告：在 {results_dir} 下没取到原文(方案 text)，表格只显示判定。"
              f"可加 --solutions-dir 指向 0907_sv 那类含 solutions.json 的目录")
    out_dir = args.out or os.path.dirname(os.path.normpath(args.verdicts_dir))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, args.out_name or (os.path.basename(os.path.normpath(out_dir)) + "_报告"))
    print(f"[report] 待渲染图片数: {len(items)} -> {base}.(pdf|pptx)")
    if not args.skip_pdf:
        render_pdf(items, base + ".pdf")
    if not args.skip_pptx:
        render_pptx(items, base + ".pptx")


if __name__ == "__main__":
    main()
