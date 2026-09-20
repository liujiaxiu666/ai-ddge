#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 qwen/gpt 评估结果（output/<xxx>_eval/raw/<图片>/verdicts.json）渲染成报告：
  - 左侧：用户原图
  - 右侧：按【方案(solution) × 维度(dim)】用表格展示 analysis / advice 的
          label(correct/wrong/partial/null) + reason，label 按颜色着色。
支持输出 PDF 与 PPTX（默认都出；可 --skip-pdf / --skip-pptx）。

用法：
  python eval_report.py --verdicts-dir /workspace/ai-ddge/7dimensions/output/0915_sv_FIRST_GOOD_PER_POOR_qwen3vlflash/gpt54_eval/raw --out /workspace/ai-ddge/7dimensions/output/0915_sv_FIRST_GOOD_PER_POOR_qwen3vlflash/gpt54_eval  --solutions_dir /workspace/ai-ddge/7dimensions/output/0915_sv_FIRST_GOOD_PER_POOR_qwen3vlflash
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
                              "左：原图  |  右： 评估（原·缺陷分析/建议 + label + reason）")
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
    """每(图片,方案)一页；表格版式与 PDF 对齐：
    列宽按比例分配、统一字号、按内容量估算行高(不高不矮且不超页)、
    浅灰细边框 + 表头底色 + 斑马纹；判定标签作为独立 run 上色加粗。"""
    from pptx import Presentation
    from pptx.util import Inches
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---- 与 PDF 对齐的表格版式参数 ----
    RIGHT_W = 8.55                       # 表格总宽(inch)，与 PDF 右侧宽度一致
    # 列宽：维度窄列 + 原文列宽(34%)、判定列窄(16%)，原文是长文本主源
    COL_W = [0.62] + [(RIGHT_W - 0.62) * r for r in (0.34, 0.16, 0.34, 0.16)]
    HEAD_SIZE, DIM_SIZE, BODY_SIZE = 9.5, 9.5, 8.5   # 表头/维度/正文 默认字号(pt)
    TABLE_LEFT, TABLE_TOP = 4.5, 0.42
    AVAIL_H = 7.5 - TABLE_TOP - 0.10     # 表格可用高度(inch)，底部只留 0.10 余量
    # 字号档位：(表头, 维度, 正文, 表头行高, 数据行最小行高, 行距倍率)
    # 行高按“含上下内边距”的宽松估算，宁高勿低，避免文字被裁；
    # 整页放不下时逐档降字号，而不是压缩行高（否则又会截字）。
    TIERS = [
        (9.5, 9.5, 8.5, 0.38, 0.32, 1.45),
        (9.0, 8.5, 7.5, 0.34, 0.26, 1.36),
        (8.5, 8.0, 6.8, 0.30, 0.22, 1.28),
    ]
    HEAD_H, MIN_ROW, LEAD = TIERS[0][3], TIERS[0][4], TIERS[0][5]

    for it in items:
        name = it.get("image", "")
        img_path = it.get("input_image", "")
        sols = group_by_solution(it.get("dims", []))
        orig_sols = it.get("_orig", {}) or {}
        for sol, dims in sols.items():
            od = orig_sols.get(sol, {})
            # 行数据：(维度, 原·缺陷分析, (label,reason)分析判定, 原·整改建议, (label,reason)建议判定)
            rows = []
            for dim in DIM_ORDER:
                if dim not in dims:
                    continue
                e = dims[dim]
                o = od.get(dim) or {}
                rows.append((
                    DIM_CN[dim],
                    (o.get("analysis") or "").strip() or "（未提供）",
                    _split_label_reason(e.get("analysis")),
                    (o.get("advice") or "").strip() or "（未提供）",
                    _split_label_reason(e.get("advice")),
                ))
            nrows = len(rows) + 1

            # 文本矩阵（用于估行高）
            texts = [["维度", "原·缺陷分析", "分析判定", "原·整改建议", "建议判定"]]
            texts += [[d, oa, ana[1], oad, adv[1]]
                      for (d, oa, ana, oad, adv) in rows]
            # 逐档尝试字号：让 估高字号 == 渲染字号，保证行高足以完整放下全部文字
            head_pt = dim_pt = body_pt = None
            for (h_pt, d_pt, b_pt, hh, mr, ld) in TIERS:
                sizes = [d_pt, b_pt, b_pt, b_pt, b_pt]
                row_h = _pptx_row_heights(texts, COL_W, sizes, hh, mr, ld)
                total_h = sum(row_h)
                if total_h <= AVAIL_H + 1e-6:
                    head_pt, dim_pt, body_pt = h_pt, d_pt, b_pt
                    break
            else:        # 最小档仍超高：等比压缩兜底（极少数超长页）
                h_pt, d_pt, b_pt, hh, mr, ld = TIERS[-1]
                head_pt, dim_pt, body_pt = h_pt, d_pt, b_pt
                sizes = [d_pt, b_pt, b_pt, b_pt, b_pt]
                row_h = _pptx_row_heights(texts, COL_W, sizes, hh, mr, ld)
                total_h = sum(row_h)
                print(f"[report] warn {name} 方案{sol}: 最小档字号仍估算 "
                      f"{total_h:.2f}in > 可用 {AVAIL_H:.2f}in，等比压缩兜底")
                k = AVAIL_H / total_h
                row_h = [h * k for h in row_h]
                total_h = AVAIL_H

            slide = prs.slides.add_slide(blank)
            # 左侧原图（统一转 JPEG 字节流，兼容 .mpo 等非常规格式）
            if img_path and os.path.exists(img_path):
                stream, iw, ih = _image_stream(img_path)
                r = min(3.9 / iw, 6.4 / ih)
                slide.shapes.add_picture(stream, Inches(0.25), Inches(0.6),
                                         width=Inches(iw * r), height=Inches(ih * r))
            # 标题
            tb = slide.shapes.add_textbox(Inches(0.25), Inches(0.06),
                                          Inches(4.05), Inches(0.5))
            _pptx_fill_paras(tb.text_frame,
                             [[(f"{name}   方案 {sol}",
                                dict(size=15, bold=True, color="1f3864"))]])

            # 右侧表格框架：先建最小高度，随后逐行设高
            gf = slide.shapes.add_table(nrows, 5, Inches(TABLE_LEFT),
                                        Inches(TABLE_TOP), Inches(RIGHT_W),
                                        Inches(min(total_h, AVAIL_H)))
            tbl = gf.table
            # 关闭默认 Office 首行高亮/斑马，改用自绘的浅色样式
            tblPr = tbl._tbl.tblPr
            tblPr.set("firstRow", "0")
            tblPr.set("bandRow", "0")
            for c, w in enumerate(COL_W):
                tbl.columns[c].width = Inches(w)
            for r, h in enumerate(row_h):
                tbl.rows[r].height = Inches(h)

            # 表头
            for c, htxt in enumerate(["维度", "原·缺陷分析", "分析判定",
                                      "原·整改建议", "建议判定"]):
                cell = tbl.cell(0, c)
                _pptx_style_cell(cell, fill="eef1f7")
                _pptx_fill_paras(cell.text_frame,
                                 [[(htxt, dict(size=head_pt, bold=True,
                                               color="1f3864"))]],
                                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
            # 数据行：斑马纹底色，标签列做成 [label]+reason 两个 run
            for i, (d, oa, ana, oad, adv) in enumerate(rows, start=1):
                bg = "ffffff" if i % 2 else "f8f9fa"
                _pptx_style_cell(tbl.cell(i, 0), fill=bg)
                _pptx_fill_paras(tbl.cell(i, 0).text_frame,
                                 [[(d, dict(size=dim_pt, bold=True,
                                            color="1f3864"))]],
                                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
                for c, txt in ((1, oa), (2, ana), (3, oad), (4, adv)):
                    cell = tbl.cell(i, c)
                    _pptx_style_cell(cell, fill=bg)
                    if c in (2, 4):
                        lab, reason = txt
                        paras = [[
                            (f"[{LABEL_CN[lab]}] ",
                             dict(size=body_pt, bold=True, color=LABEL_COLOR[lab])),
                            (reason, dict(size=body_pt, color="24292f")),
                        ]]
                    else:
                        paras = [[(txt, dict(size=body_pt, color="24292f"))]]
                    _pptx_fill_paras(cell.text_frame, paras,
                                     align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
            # 最后统一画浅灰细边框（覆盖默认样式影响）
            for r in range(nrows):
                for c in range(5):
                    _pptx_border_cell(tbl.cell(r, c))
    prs.save(out_path)
    print(f"[report] PPTX: {out_path}")


# ---------------------------------------------------------------------------
# PPTX 排版辅助（与 PDF 版式对齐）
# ---------------------------------------------------------------------------
FONT_CN = "Microsoft YaHei"


def _split_label_reason(e: dict):
    """analysis/advice dict -> (label, reason)，统一缺省值。"""
    if not e:
        return "null", "（无理由）"
    label = e.get("label") or "null"
    label = label if label in LABEL_CN else "null"
    reason = (e.get("reason") or "").strip() or "（无理由）"
    return label, reason


def _pptx_est_lines(text: str, width_in: float, font_pt: float) -> int:
    """估算文本在 列宽×字号 下占几行：全角≈1em，半角≈0.55em（偏保守）。"""
    import math
    usable_pt = max(0.1, width_in - 0.14) * 72.0
    ems_per_line = max(1.0, usable_pt / font_pt)
    lines = 0
    for seg in str(text).split("\n"):
        ems = sum(1.0 if ord(ch) > 0x2E7F else 0.55 for ch in seg)
        lines += max(1, math.ceil(ems / ems_per_line))
    return lines


def _pptx_row_heights(texts, col_w, sizes, head_h, min_row, lead):
    """按每格文本量估算各行高(inch)。每行高度 = 文字所需 + 上下内边距(PAD)，宁高勿低。"""
    pad = 0.06                       # 单元格上下内边距(0.04) + 安全余量
    row_h = [head_h + pad]
    for r in range(1, len(texts)):
        h = min_row
        for c, txt in enumerate(texts[r]):
            n_lines = _pptx_est_lines(txt, col_w[c], sizes[c])
            h = max(h, n_lines * sizes[c] * lead / 72.0)
        row_h.append(h + pad)
    return row_h


def _pptx_style_run(run, sty: dict):
    """统一 run 的字号/粗细/颜色/中英文字体，避免不同机器字形与默认 18pt 不一致。"""
    from pptx.util import Pt
    from pptx.dml.color import RGBColor
    from pptx.oxml.ns import qn
    f = run.font
    f.size = Pt(sty.get("size", 9))
    f.bold = bool(sty.get("bold", False))
    f.name = FONT_CN                        # 会生成 a:latin
    f.color.rgb = RGBColor(*_hex(sty.get("color", "24292f")))
    rPr = run._r.get_or_add_rPr()
    latin = rPr.find(qn("a:latin"))
    ea = rPr.find(qn("a:ea"))
    cs = rPr.find(qn("a:cs"))
    if latin is not None:
        if ea is None:
            ea = latin.makeelement(qn("a:ea"), {"typeface": FONT_CN})
            latin.addnext(ea)
        else:
            ea.set("typeface", FONT_CN)
        anchor = ea if ea is not None else latin
        if cs is None:
            cs = anchor.makeelement(qn("a:cs"), {"typeface": FONT_CN})
            anchor.addnext(cs)
        else:
            cs.set("typeface", FONT_CN)


def _pptx_fill_paras(tf, paras, align=None, anchor=None):
    """填充 text_frame。paras = [[(text, style_dict), ...], ...]（每段可多 run）。"""
    from pptx.enum.text import MSO_ANCHOR
    tf.word_wrap = True
    if anchor is not None:
        tf.vertical_anchor = anchor
    tf.clear()
    for idx, runs in enumerate(paras):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        if align is not None:
            p.alignment = align
        p.line_spacing = 1.0
        for text, sty in runs:
            run = p.add_run()
            run.text = text
            _pptx_style_run(run, sty)


def _pptx_style_cell(cell, fill: str):
    """单元格底色与内边距（显式填充以盖掉默认表格样式）。"""
    from pptx.util import Inches
    from pptx.dml.color import RGBColor
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(*_hex(fill))
    cell.margin_left = Inches(0.05)
    cell.margin_right = Inches(0.05)
    cell.margin_top = Inches(0.02)
    cell.margin_bottom = Inches(0.02)


def _pptx_border_cell(cell, color="d0d7de", width_emu=9525):
    """给单元格四边画 0.75pt 细边框；按 OOXML 顺序插到填充之前。"""
    from pptx.oxml.ns import qn
    tcPr = cell._tc.get_or_add_tcPr()
    tags = [qn("a:lnL"), qn("a:lnR"), qn("a:lnT"), qn("a:lnB")]
    for tag in tags:
        for old in tcPr.findall(tag):
            tcPr.remove(old)
    first = None
    for child in tcPr:
        if child.tag not in tags:
            first = child
            break
    for tag in tags:
        ln = tcPr.makeelement(tag, {"w": str(width_emu), "cap": "flat",
                                    "cmpd": "sng", "algn": "ctr"})
        fill_el = ln.makeelement(qn("a:solidFill"), {})
        clr = fill_el.makeelement(qn("a:srgbClr"), {"val": color})
        fill_el.append(clr)
        ln.append(fill_el)
        if first is not None:
            first.addprevious(ln)
        else:
            tcPr.append(ln)


def _hex(h):
    """'#rrggbb' 或 'rrggbb' -> (r, g, b) 字节。兼容带/不带 # 的写法。"""
    h = h.lstrip("#").strip()
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


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
    ap.add_argument("--solutions_dir", default=None)
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
