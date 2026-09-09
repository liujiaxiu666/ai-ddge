#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 7dimensions 测试结果生成 PPT(pptx) + PDF 报告。

数据来源：测试结果目录下每个「图片名文件夹」里的 solutions.json，
其中包含 input_image（测试原图）、evidence_by_dim（每维检索到的 poor 图 / 放进
prompt 的 good 图 / dim_original_text）以及 solutions（每个 direction 方案的 text）。

版式（16:9，每个方案 2 页，左侧始终为测试原图）：
  第 1 页：右侧从上到下 5 行（维度 1~5：ratio/comp/camera/position/pose），
           每行从左到右 = poor图 + good图 + [该维指导文本 + 拍照缺陷分析]
  第 2 页：右侧上 2 行（维度 6~7：focus/color）同上；
           底部 = 完整“针对性整改建议”（左）+ 该方案生成图（右，放大）

适配数据（每方案每维度 1 图 + 1 文）：
  方案 k 每个维度取检索结果第 k 条证据（1 张 poor + 1 张 good + 1 条指导文本）；
  每方案展示其生成图（solutions.json 的 edit_image / solution_N.png）。

用法（整批 -> 单份报告，保存到输入测试路径）：
  python make_ppt_0901.py \
      --results-dir /workspace/ai-ddge/7dimensions/output/0907_sv \
      --save-dir   /workspace/ai-ddge/7dimensions/output/0907_sv \
      --data-root  /workspace/ai-ddge/7dimensions/dataset
  # 可选：--only "669da389b6114a208175994fa13af3fe" 只处理某一张
  #       --max-solutions 3 每张只取前 N 个方案
  #       --out-name "0830报告" 自定义输出文件名（默认=<结果目录名>_报告）
说明：
  - 整个结果目录的所有图片会合成「单份」pptx + pdf（不是每张一个）。
  - 每个方案每维度只取第 k 条证据（1 poor + 1 good + 1 指导文本），适配新数据构成。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile

from PIL import Image

# ---------------------------------------------------------------------------
# 依赖检查
# ---------------------------------------------------------------------------
_HAS_PPTX = True
_HAS_RL = True
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.oxml.ns import qn
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN
except Exception:  # pragma: no cover
    _HAS_PPTX = False

try:
    from reportlab.pdfgen import canvas as _rl_canvas
    from reportlab.lib.pagesizes import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.lib.utils import ImageReader
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
except Exception as e:  # pragma: no cover
    print("[warn] reportlab 不可用，将无法导出 PDF:", e)
    _HAS_RL = False

# ---------------------------------------------------------------------------
# 布局常量（单位：英寸；16:9）
# ---------------------------------------------------------------------------
PAGE_W = 13.333
PAGE_H = 7.5
MARGIN = 0.25
TITLE_Y = 0.10
TITLE_H = 0.55
CONTENT_Y = 0.85
CONTENT_H = PAGE_H - CONTENT_Y - 0.30          # 6.35
LEFT_X = MARGIN
LEFT_W = 4.0
RIGHT_X = MARGIN + LEFT_W + 0.30               # 4.55
RIGHT_W = PAGE_W - MARGIN - RIGHT_X            # ~8.53
BLOCK_GAP = 0.12
HEAD_H = 0.26
CAP_H = 0.14
PAGE4_TOP_H = 3.3
TEXT_BASE = 9.0
TEXT_MIN = 6.5

BLACK = (0.0, 0.0, 0.0)
GRAY = (0.42, 0.42, 0.42)
DARKBLUE = (0.12, 0.25, 0.42)

DIM_ORDER = ["ratio", "composition", "camera", "position", "pose", "focus", "color"]
LABEL = {
    "ratio": "1 · Ratio 画面比例",
    "composition": "2 · Composition 构图取景",
    "camera": "3 · Camera 机位视角",
    "position": "4 · Position 主体位置",
    "pose": "5 · Pose 姿态动作",
    "focus": "6 · Focus 对焦景深",
    "color": "7 · Color 色彩光影",
}
CN = {
    "ratio": "画面比例", "composition": "构图取景", "camera": "机位视角",
    "position": "主体位置", "pose": "姿态动作", "focus": "对焦景深",
    "color": "色彩光影",
}
EVIDENCE_PER_DIM = 0          # 0 = 展示该维全部证据（所有 poor/good 对）；>0 只取前 N 条
DEFAULT_DATA_ROOTS = ["/workspace/ai-ddge/7dimensions/dataset"]

# 多证据网格布局参数（英寸）：优先保证图片尽量大，文字适当缩小让位给图片
PAIR_GAP = 0.12      # 同一条证据内 poor 与 good 的间距
COL_GAP = 0.12       # 网格列间距
ROW_GAP = 0.12       # 网格行间距
PAD = 0.10           # 维度块内边距
# 图片盒尺寸递退档（宽, 高）：从大到小，证据多时逐档缩小；首档明显放大提升可读性
IMG_STEPS = [(1.35, 1.10), (1.20, 1.00), (1.05, 0.88), (0.90, 0.75), (0.75, 0.62)]
FONT_STEPS = [8.0, 7.0, 6.0, 5.5]   # 文字适当缩小，把更多高度让给图片

# ---- 每方案 2 页版式参数 ----
ROW_GAP2 = 0.10             # 行间距
IMG_W = 1.35                # poor/good 图盒宽
IMG_CAP_H = 0.13            # 图注高度
GEN_W2 = 3.2                # 第 2 页生成图放大宽度（建议区右侧）
PAGE1_DIMS = DIM_ORDER[:5]           # 第 1 页右侧 5 行：ratio..pose
PAGE2_TOP_DIMS = DIM_ORDER[5:]       # 第 2 页右上 2 行：focus, color

# ---------------------------------------------------------------------------
# 解析 solutions.json
# ---------------------------------------------------------------------------
_DIM_RE = (r"^(画面比例|构图取景|机位视角|主体位置|姿态动作|姿态|对焦景深|对焦|色彩光影|色彩)"
           r"\s*(?:整改)?\s*[：:]\s*(.*)$")
_DIM_NAME2KEY = {
    "画面比例": "ratio", "构图取景": "composition", "机位视角": "camera",
    "主体位置": "position", "姿态动作": "pose", "姿态": "pose",
    "对焦景深": "focus", "对焦": "focus", "色彩光影": "color", "色彩": "color",
}


def _split_sections(text: str):
    """拆出「拍照缺陷分析」和「针对性整改建议」两个段落。"""
    m_def = text.find("# 拍照缺陷分析")
    m_adv = text.find("# 针对性整改建议")
    defect, advice = "", ""
    if m_def != -1 and (m_adv > m_def or m_adv == -1):
        defect = text[m_def + len("# 拍照缺陷分析"): (m_adv if m_adv > m_def else len(text))]
    if m_adv != -1:
        advice = text[m_adv + len("# 针对性整改建议"):]
    return defect, advice


def _parse_bullets(section: str) -> dict:
    """把段落里的 '- 画面比例：xxx' 类条目解析成 {dim_key: 文本}。"""
    out = {}
    if not section:
        return out
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        body = line[1:].strip()
        body = re.sub(r"^[（(][^）)]*[）)]\s*", "", body)   # 去掉“（仅使用本组证据）”
        m = re.match(_DIM_RE, body)
        if m:
            key = _DIM_NAME2KEY.get(m.group(1))
            if key:
                txt = m.group(2).strip()
                # 剥掉冒号后紧跟的“（仅使用本组证据）”等括号说明
                txt = re.sub(r"^[（(][^）)]*[）)]\s*", "", txt)
                out[key] = txt
    return out


def parse_solution_text(text: str) -> dict:
    defect, advice = _split_sections(text)
    return {"defect": _parse_bullets(defect), "advice": _parse_bullets(advice)}


# ---------------------------------------------------------------------------
# 图片解析（多根路径回退）
# ---------------------------------------------------------------------------
def resolve_image(path: str, roots: list) -> str:
    """按多种策略解析图片真实路径，找不到返回原路径（渲染时显示“图片缺失”）。"""
    if not path:
        return path
    cands = [path]
    for root in roots:
        if root:
            cands.append(os.path.join(root, path.lstrip("/")))
    cands.append(path.replace("/workspace/ai-ddge/AesRecon",
                              "/workspace/ai-ddge/7dimensions/dataset/AesRecon"))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return path


# ---------------------------------------------------------------------------
# 文本排版工具
# ---------------------------------------------------------------------------
_FULLWIDTH_RANGES = ((0x3000, 0x303F), (0xFF00, 0xFFEF))
_FULLWIDTH_SINGLE = {0x00B7, 0x2018, 0x2019, 0x201C, 0x201D, 0x2013, 0x2014, 0x2026}


def _char_w(ch: str, pt: float) -> float:
    """近似字符宽度（pt）：CJK/全角标点=1.0 字号，ASCII≈0.5 字号。"""
    o = ord(ch)
    if o > 0x2E7F:
        return pt
    for a, b in _FULLWIDTH_RANGES:
        if a <= o <= b:
            return pt
    if o in _FULLWIDTH_SINGLE:
        return pt
    return pt * 0.50


def wrap_text(text: str, width_in: float, pt: float) -> list:
    width_pt = max(width_in * 72.0, 1.0)
    lines, cur, cur_w = [], "", 0.0
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur, cur_w = "", 0.0
            continue
        w = _char_w(ch, pt)
        if cur and cur_w + w > width_pt:
            lines.append(cur)
            cur, cur_w = ch, w
        else:
            cur += ch
            cur_w += w
    if cur:
        lines.append(cur)
    return lines


def needed_height(text: str, width_in: float, pt: float) -> float:
    return len(wrap_text(text, width_in, pt)) * pt * 1.25 / 72.0


def fit_pt(text: str, width_in: float, height_in: float,
           base: float = TEXT_BASE, min_pt: float = TEXT_MIN) -> float:
    h0 = needed_height(text, width_in, base)
    if h0 <= height_in:
        return base
    pt = max(min_pt, base * height_in / max(h0, 1e-6))
    while pt > min_pt and needed_height(text, width_in, pt) > height_in:
        pt -= 0.5
    return max(min_pt, pt)


def fit_box(iw, ih, bw, bh):
    if iw <= 0 or ih <= 0:
        return (bw, bh)
    r = min(bw / iw, bh / ih)
    return (iw * r, ih * r)


# ---------------------------------------------------------------------------
# 单页元素构建
# ---------------------------------------------------------------------------
def _solution_evidence(sol, dim):
    """方案 k 的该维证据 = 检索结果第 k 条（1 张 poor + 1 张 good + 1 条指导文本）。"""
    evs = sol["ev"].get(dim) or []
    k = sol["index"]
    if k - 1 < len(evs):
        return evs[k - 1]
    return evs[-1] if evs else {"poor": "", "good": "", "text": ""}


def _img_fit(path, bw, bh, dims_cache):
    """把图片按比例放入 (bw, bh) 盒子，返回实际 (iw, ih)；尺寸按原图缓存。"""
    if not path or not os.path.exists(path):
        return bw, bh
    key = (path, "size")
    if key not in dims_cache:
        try:
            with Image.open(path) as im:
                dims_cache[key] = im.size
        except Exception:
            dims_cache[key] = (1, 1)
    iw, ih = dims_cache[key]
    return fit_box(iw, ih, bw, bh)


def _dim_row_items2(x, y, w, h, ev, label, sol, dim, roots, dims_cache):
    """每方案 2 页版式的维度行：标题 + poor图 + good图 + [指导文本+拍照缺陷分析]（无生成图）。"""
    items = []
    label_h = 0.22
    items.append({"t": "text", "x": x, "y": y, "w": w, "h": label_h,
                  "text": f"【{label}】", "size": 11, "bold": True, "color": DARKBLUE})

    content_h = h - label_h - 0.04
    cy = y + label_h
    img_h = max(content_h - IMG_CAP_H, 0.2)
    gap = 0.10

    # ---- 左侧 poor/good 图 ----
    for side, p, cx in (("poor", ev.get("poor") or "", x),
                        ("good", ev.get("good") or "", x + IMG_W + gap)):
        iw, ih = _img_fit(p, IMG_W, img_h, dims_cache)
        ix = cx + (IMG_W - iw) / 2
        iy = cy + (img_h - ih) / 2
        items.append({"t": "image", "x": ix, "y": iy, "w": iw, "h": ih, "path": p})
        items.append({"t": "text", "x": cx, "y": cy + img_h, "w": IMG_W, "h": IMG_CAP_H,
                      "text": "poor" if side == "poor" else "good",
                      "size": 7, "bold": False, "color": GRAY})

    # ---- 右侧：指导文本 + 拍照缺陷分析（放一起，占满整行剩余宽度）----
    text_x = x + 2 * IMG_W + 2 * gap
    text_w = w - (text_x - x) - gap
    txt = ev.get("text") or "【该维度无检索证据】"
    ana = sol["defect"].get(dim, "")
    if ana:
        txt = f"{txt}\n【拍照缺陷分析】{ana}"
    pt = fit_pt(txt, text_w, content_h, base=8.0, min_pt=5.0)
    items.append({"t": "text", "x": text_x, "y": cy, "w": text_w, "h": content_h,
                  "text": txt, "size": pt, "bold": False, "color": BLACK})
    return items


def _advice_image_block_items(x, y, w, h, sol, gen_img, roots, dims_cache):
    """第 2 页底部：左侧 = 完整“针对性整改建议”（2 列），右侧 = 该方案生成图（放大）。"""
    items = []
    image_w = GEN_W2
    gap = 0.2
    adv_w = w - image_w - gap

    # ---- 左侧：完整整改建议背景框 + 2 列文本 ----
    items.append({"t": "rect", "x": x - 0.05, "y": y - 0.02, "w": adv_w + 0.10,
                  "h": h - 0.02, "fill": (0.965, 0.985, 1.0), "line": (0.55, 0.72, 0.85)})
    lines = []
    for d in DIM_ORDER:
        adv = (sol["advice"].get(d) or "").strip()
        if adv:
            lines.append(f"- {CN[d]}整改：{adv}")
    if not lines:
        lines = ["【该方案暂无针对性整改建议】"]
    col_n = math.ceil(len(lines) / 2)
    col_gap = 0.15
    col_w = (adv_w - col_gap) / 2
    for col, start in enumerate((0, col_n)):
        grp = lines[start:start + col_n]
        if not grp:
            continue
        cx = x + col * (col_w + col_gap)
        txt = "\n".join(grp)
        pt = fit_pt(txt, col_w - 0.10, h - 0.12, base=9.0, min_pt=5.5)
        items.append({"t": "text", "x": cx, "y": y + 0.05, "w": col_w - 0.10,
                      "h": h - 0.12, "text": txt, "size": pt,
                      "bold": False, "color": BLACK})

    # ---- 右侧：该方案生成图（放大）----
    ix = x + adv_w + gap
    iw_area, ih_area = image_w, h - 0.2
    giw, gih = _img_fit(gen_img, iw_area, ih_area, dims_cache)
    gix = ix + (iw_area - giw) / 2
    giy = y + 0.1 + (ih_area - gih) / 2
    items.append({"t": "image", "x": gix, "y": giy, "w": giw, "h": gih, "path": gen_img})
    items.append({"t": "text", "x": ix, "y": y + h - 0.24, "w": iw_area, "h": 0.2,
                  "text": "方案生成图", "size": 8, "bold": True, "color": DARKBLUE})
    return items


def build_slides(data, roots):
    """data -> list[slide]；每个方案 2 页（左侧始终为测试原图）：
    第 1 页：右侧 5 行（维度 1~5），每行 = poor图 + good图 + [指导文本+拍照缺陷分析]；
    第 2 页：右侧上 2 行（维度 6~7）同上，底部 = 完整整改建议（左）+ 生成图（右，放大）。
    """
    slides = []
    dims_cache = {}
    orig = data["input_image"]
    try:
        with Image.open(orig) as im:
            ow, oh = im.size
    except Exception:
        ow, oh = 800, 600
    osw, osh = fit_box(ow, oh, LEFT_W - 0.1, CONTENT_H)
    ox = LEFT_X + ((LEFT_W - 0.1) - osw) / 2
    oy = CONTENT_Y + (CONTENT_H - osh) / 2

    row_h = (CONTENT_H - 4 * ROW_GAP2) / 5
    for sol in data["solutions"]:
        base_title = f"方案{sol['index']} · {sol['direction']}"
        gen_img = sol.get("gen_img") or ""
        orig_items = [
            {"t": "image", "x": ox, "y": oy, "w": osw, "h": osh, "path": orig},
            {"t": "text", "x": LEFT_X, "y": CONTENT_Y + CONTENT_H - 0.24,
             "w": LEFT_W, "h": 0.22,
             "text": f"测试原图：{data['name']}", "size": 8, "bold": False, "color": GRAY},
        ]

        # ---- 第 1 页：右侧 5 行（维度 1~5）----
        title1 = {"t": "text", "x": MARGIN, "y": TITLE_Y, "w": PAGE_W - 2 * MARGIN,
                  "h": TITLE_H, "size": 15, "bold": True, "color": BLACK,
                  "text": f"{base_title}    第 1/2 页 · 各维度证据（1~5 维）"}
        items1 = [title1] + list(orig_items)
        ry = CONTENT_Y
        for dim in PAGE1_DIMS:
            items1 += _dim_row_items2(RIGHT_X, ry, RIGHT_W, row_h,
                                      _solution_evidence(sol, dim), LABEL[dim],
                                      sol, dim, roots, dims_cache)
            ry += row_h + ROW_GAP2
        slides.append(items1)

        # ---- 第 2 页：右上 2 行（维度 6~7）+ 底部 建议+生成图 ----
        title2 = {"t": "text", "x": MARGIN, "y": TITLE_Y, "w": PAGE_W - 2 * MARGIN,
                  "h": TITLE_H, "size": 15, "bold": True, "color": BLACK,
                  "text": f"{base_title}    第 2/2 页 · 6~7 维证据 + 完整整改建议 + 生成图"}
        items2 = [title2] + list(orig_items)
        ry = CONTENT_Y
        for dim in PAGE2_TOP_DIMS:
            items2 += _dim_row_items2(RIGHT_X, ry, RIGHT_W, row_h,
                                      _solution_evidence(sol, dim), LABEL[dim],
                                      sol, dim, roots, dims_cache)
            ry += row_h + ROW_GAP2
        adv_h = 3 * row_h + 2 * ROW_GAP2
        items2 += _advice_image_block_items(RIGHT_X, ry, RIGHT_W, adv_h,
                                            sol, gen_img, roots, dims_cache)
        slides.append(items2)
    return slides


# ---------------------------------------------------------------------------
# PPTX 渲染
# ---------------------------------------------------------------------------
def _set_font(run, name, size, bold, color):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", name)


def render_pptx(slides_items, out_path, font_name="Microsoft YaHei"):
    if not _HAS_PPTX:
        raise RuntimeError("缺少 python-pptx，无法生成 pptx")
    prs = Presentation()
    prs.slide_width = Inches(PAGE_W)
    prs.slide_height = Inches(PAGE_H)
    blank = prs.slide_layouts[6]
    for items in slides_items:
        slide = prs.slides.add_slide(blank)
        for it in items:
            x, y, w, h = (Inches(it["x"]), Inches(it["y"]),
                          Inches(it["w"]), Inches(it["h"]))
            if it["t"] == "image":
                p = it["path"]
                if p and os.path.exists(p):
                    try:
                        slide.shapes.add_picture(p, x, y, w, h)
                        continue
                    except Exception:
                        pass
                sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
                sh.fill.solid()
                sh.fill.fore_color.rgb = RGBColor(0xE6, 0xE6, 0xE6)
                sh.line.fill.background()
                tf = sh.text_frame
                tf.word_wrap = True
                p_ = tf.paragraphs[0]
                r_ = p_.add_run()
                r_.text = "图片缺失"
                _set_font(r_, font_name, 9, False, (0.3, 0.3, 0.3))
            elif it["t"] == "rect":
                sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
                sh.fill.solid()
                sh.fill.fore_color.rgb = RGBColor(int(it["fill"][0] * 255),
                                                  int(it["fill"][1] * 255),
                                                  int(it["fill"][2] * 255))
                if it.get("line"):
                    sh.line.color.rgb = RGBColor(int(it["line"][0] * 255),
                                                 int(it["line"][1] * 255),
                                                 int(it["line"][2] * 255))
                    sh.line.width = Pt(0.75)
                else:
                    sh.line.fill.background()
                sh.shadow.inherit = False
            elif it["t"] == "text":
                tb = slide.shapes.add_textbox(x, y, w, h)
                tf = tb.text_frame
                tf.word_wrap = True
                tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
                lines = wrap_text(it["text"], it["w"], it["size"])
                for i, ln in enumerate(lines):
                    p_ = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    p_.alignment = PP_ALIGN.LEFT
                    r_ = p_.add_run()
                    r_.text = ln
                    _set_font(r_, font_name, it["size"], it.get("bold", False),
                              it.get("color", BLACK))
    prs.save(out_path)


# ---------------------------------------------------------------------------
# PDF 渲染（reportlab 直渲，无需 LibreOffice）
# ---------------------------------------------------------------------------
def render_pdf(slides_items, out_path):
    if not _HAS_RL:
        raise RuntimeError("缺少 reportlab，无法生成 PDF")
    c = _rl_canvas.Canvas(out_path, pagesize=(PAGE_W * inch, PAGE_H * inch))
    _img_cache = {}

    def _ir(path):
        if path not in _img_cache:
            _img_cache[path] = ImageReader(path)
        return _img_cache[path]

    for items in slides_items:
        for it in items:
            if it["t"] == "image":
                p = it["path"]
                px, py = it["x"] * inch, (PAGE_H - it["y"] - it["h"]) * inch
                pw, ph = it["w"] * inch, it["h"] * inch
                if p and os.path.exists(p):
                    try:
                        c.drawImage(_ir(p), px, py, pw, ph, mask="auto")
                        continue
                    except Exception:
                        pass
                c.setFillColorRGB(0.9, 0.9, 0.9)
                c.rect(px, py, pw, ph, fill=1, stroke=0)
                c.setFillColorRGB(0.35, 0.35, 0.35)
                c.setFont("STSong-Light", 8)
                c.drawCentredString(px + pw / 2, py + ph / 2, "图片缺失")
            elif it["t"] == "rect":
                px, py = it["x"] * inch, (PAGE_H - it["y"] - it["h"]) * inch
                pw, ph = it["w"] * inch, it["h"] * inch
                c.setFillColorRGB(*it["fill"])
                if it.get("line"):
                    c.setStrokeColorRGB(*it["line"])
                    c.setLineWidth(0.75)
                    c.rect(px, py, pw, ph, fill=1, stroke=1)
                else:
                    c.rect(px, py, pw, ph, fill=1, stroke=0)
            elif it["t"] == "text":
                size = it["size"]
                lines = wrap_text(it["text"], it["w"], size)
                x0 = it["x"] * inch
                y = (PAGE_H - it["y"]) * inch - size
                leading = size * 1.25
                c.setFont("STSong-Light", size)
                c.setFillColorRGB(*it.get("color", BLACK))
                for ln in lines:
                    c.drawString(x0, y, ln)
                    y -= leading
        c.showPage()
    c.save()


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
def _find_gen_image(folder, s) -> str:
    """查找该方案的生成图：优先 solutions.json 的 edit_image，回退 folder/solution_{index}.png。"""
    idx = s.get("index", 0)
    cands = [(s.get("edit_image") or "").strip()]
    if cands[0]:
        cands.append(os.path.join(folder, os.path.basename(cands[0])))
    cands.append(os.path.join(folder, f"solution_{idx}.png"))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return ""


def load_solution_data(folder, roots) -> dict:
    js_path = os.path.join(folder, "solutions.json")
    with open(js_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    name = os.path.basename(os.path.normpath(folder))
    ev_by_dim = {}
    for dim in DIM_ORDER:
        evs = d.get("evidence_by_dim", {}).get(dim, []) or []
        if EVIDENCE_PER_DIM > 0:
            evs = evs[:EVIDENCE_PER_DIM]
        ev_by_dim[dim] = [{
            "poor": resolve_image(e.get("poor_image_path", ""), roots),
            "good": resolve_image(e.get("good_image_path", ""), roots),
            "text": e.get("dim_original_text", ""),
        } for e in evs]

    solutions = []
    for s in d.get("solutions", []):
        parsed = parse_solution_text(s.get("text", ""))
        solutions.append({
            "index": s.get("index", 0),
            "direction": s.get("direction", ""),
            "defect": parsed["defect"],
            "advice": parsed["advice"],
            "gen_img": resolve_image(_find_gen_image(folder, s), roots),
            "ev": ev_by_dim,
        })
    return {
        "name": name,
        "input_image": resolve_image(d.get("input_image", ""), roots),
        "solutions": solutions,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _thumb_path(path, thumb_dir, max_side=480):
    """图片降采样到 max_side 内（同一原图只生成一次），用于控制 pptx/pdf 体积。"""
    if not path or not os.path.exists(path):
        return path
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:16]
    out = os.path.join(thumb_dir, h + ".jpg")
    if not os.path.exists(out):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((max_side, max_side))
                im.save(out, "JPEG", quality=82)
        except Exception:
            return path
    return out


def _prepare_thumbs(slides_items):
    """把所有图片路径换成降采样缩略图，返回临时缩略图目录（用后由调用方清理）。"""
    thumb_dir = tempfile.mkdtemp(prefix="ddge_thumbs_")
    for items in slides_items:
        for it in items:
            if it["t"] == "image" and it.get("path"):
                it["path"] = _thumb_path(it["path"], thumb_dir)
    return thumb_dir


def main():
    ap = argparse.ArgumentParser(description="7dimensions 测试结果 -> 单份 PPT + PDF（含全部证据）")
    ap.add_argument("--results-dir", required=True, help="测试结果目录（含各图片名文件夹/solutions.json）")
    ap.add_argument("--save-dir", default=None, help="保存目录（默认=--results-dir，即输入的测试路径）")
    ap.add_argument("--data-root", action="append", default=[],
                    help="数据/图片根目录（可多次指定，用于解析 poor/good/原图缺失路径）")
    ap.add_argument("--only", default=None, help="只处理含该关键字的图片文件夹")
    ap.add_argument("--max-solutions", type=int, default=0, help="每张只取前 N 个方案（0=全部）")
    ap.add_argument("--out-name", default=None,
                    help="输出文件名（不含扩展名，默认=<结果目录名>_报告）")
    ap.add_argument("--skip-pptx", action="store_true", help="不生成 pptx")
    ap.add_argument("--skip-pdf", action="store_true", help="不生成 pdf")
    args = ap.parse_args()

    roots = list(args.data_root) or list(DEFAULT_DATA_ROOTS)
    save_dir = args.save_dir or args.results_dir
    os.makedirs(save_dir, exist_ok=True)

    folders = sorted(
        d for d in glob.glob(os.path.join(args.results_dir, "*", ""))
        if os.path.isfile(os.path.join(d, "solutions.json")))
    if args.only:
        folders = [d for d in folders if args.only in os.path.basename(os.path.normpath(d))]

    if not folders:
        print(f"[make] 在 {args.results_dir} 下没找到含 solutions.json 的文件夹")
        return

    print(f"[make] 待处理图片文件夹: {len(folders)} 个 -> 生成单份报告到 {save_dir}")
    all_slides = []
    for i, folder in enumerate(folders, 1):
        name = os.path.basename(os.path.normpath(folder))
        data = load_solution_data(folder, roots)
        if args.max_solutions > 0:
            data["solutions"] = data["solutions"][:args.max_solutions]
        slides = build_slides(data, roots)
        all_slides += slides
        print(f"[make] ({i}/{len(folders)}) {name}: {len(slides)} 页")

    out_base = os.path.join(
        save_dir,
        args.out_name or (os.path.basename(os.path.normpath(args.results_dir)) + "_报告"))
    print(f"[make] 共 {len(all_slides)} 页，开始渲染 ...")
    thumb_dir = _prepare_thumbs(all_slides)
    try:
        if not args.skip_pptx:
            render_pptx(all_slides, out_base + ".pptx")
        if not args.skip_pdf:
            render_pdf(all_slides, out_base + ".pdf")
    finally:
        shutil.rmtree(thumb_dir, ignore_errors=True)
    print(f"[make] 完成：{len(folders)} 张图片 -> {out_base}.pptx / .pdf（共 {len(all_slides)} 页）")


if __name__ == "__main__":
    main()
