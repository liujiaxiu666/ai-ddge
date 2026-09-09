# -*- coding: utf-8 -*-
"""
FAISS 并行检索后端（默认检索实现）
====================================
方案：每个维度一个独立的 faiss IndexFlatIP（暴力精确检索，无近似损失），
      建好后 search 走 faiss 内部 OpenMP 多线程（7 维全部检索约 1.9s）。
索引由离线重建（main.py offline / get_7dimensions.py）直接产出到 FAISS_INDEX_DIR。

对外接口：
  search_maxsim(query_patch_set, dim, topk=..., ann_topk=...,
                score_threshold=..., min_coverage=...) -> evidence list
"""
from __future__ import annotations

import os
import pickle
import shutil
from collections import defaultdict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import faiss

from config import (AGG_POOL_TOPK, ANN_TOPK, DIM_ORDER, FAISS_INDEX_DIR,
                    FAISS_NUM_THREADS, FAISS_SQ8_DIR, MAXSIM_MIN_COVERAGE,
                    MAXSIM_SCORE_THRESHOLD)


class FaissIndex:
    """每维度一个 faiss IndexFlatIP + 行级元数据；懒加载 + 缓存。"""

    def __init__(self, index_dir: str = FAISS_INDEX_DIR):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self._indices: dict = {}
        self._metas: dict = {}
        # faiss 内部 OpenMP 并行度（默认 64 = 全核）
        faiss.omp_set_num_threads(FAISS_NUM_THREADS)

    # ------------------------------------------------------------------
    # 构建（推荐）：直接从内存 buffer 写 faiss（离线重建产出）。
    # buffers[dim] = {'vecs': [np.ndarray[N,D], ...], 'meta': [tuple, ...]}
    # part_tag 非空时写成 {dim}.{part_tag}.index（供并行分段重建后合并）。
    # ------------------------------------------------------------------
    def build_from_buffers(self, buffers: dict, part_tag: str = "") -> None:
        tag = f".{part_tag}" if part_tag else ""
        for dim in DIM_ORDER:
            buf = buffers.get(dim)
            if not buf or not buf.get("vecs"):
                print(f"[faiss] {dim}{tag} 无数据，跳过", flush=True)
                continue
            vecs = np.concatenate(buf["vecs"], axis=0).astype(np.float32)
            meta = [m for row in buf["meta"] for m in row]
            idx = faiss.IndexFlatIP(vecs.shape[1])
            idx.add(vecs)
            faiss.write_index(idx, os.path.join(self.index_dir, f"{dim}{tag}.index"))
            with open(os.path.join(self.index_dir, f"{dim}{tag}.pkl"), "wb") as f:
                pickle.dump(meta, f)
            print(f"[faiss] {dim}{tag} {vecs.shape[0]} 行 -> {self.index_dir}/{dim}{tag}.index",
                  flush=True)
            del vecs

    # ------------------------------------------------------------------
    # 合并分段重建的索引（并行 4 卡 -> 每卡写 {dim}.{part}.index，这里合成最终 {dim}.index）
    # ------------------------------------------------------------------
    @staticmethod
    def merge_parts(index_dir: str = FAISS_INDEX_DIR, n_parts: int = 4,
                    dims: list = DIM_ORDER) -> None:
        for dim in dims:
            metas: list = []
            merged = None
            for p in range(n_parts):
                ip = os.path.join(index_dir, f"{dim}.{p}.index")
                if not os.path.exists(ip):
                    continue
                part = faiss.read_index(ip)
                if merged is None:
                    merged = faiss.IndexFlatIP(part.d)
                merged.merge_from(part)
                with open(os.path.join(index_dir, f"{dim}.{p}.pkl"), "rb") as f:
                    metas.extend(pickle.load(f))
                os.remove(ip)
                os.remove(os.path.join(index_dir, f"{dim}.{p}.pkl"))
            if merged is None:
                print(f"[faiss-merge] {dim} 无分段，跳过", flush=True)
                continue
            faiss.write_index(merged, os.path.join(index_dir, f"{dim}.index"))
            with open(os.path.join(index_dir, f"{dim}.pkl"), "wb") as f:
                pickle.dump(metas, f)
            print(f"[faiss-merge] {dim}: 合并 {len(metas)} 行 -> {dim}.index", flush=True)

    # ------------------------------------------------------------------
    # 转换：fp32 索引 -> SQ8 量化版（体积 4 倍缩小，加载快 4 倍）
    # 只量化索引侧向量，保留原 fp32 索引文件不删；meta(pkl) 原样复制。
    # ------------------------------------------------------------------
    @staticmethod
    def convert_to_sq8(src_dir: str = FAISS_INDEX_DIR,
                       out_dir: str = FAISS_SQ8_DIR,
                       dims: list = DIM_ORDER) -> None:
        os.makedirs(out_dir, exist_ok=True)
        for dim in dims:
            ip = os.path.join(src_dir, f"{dim}.index")
            if not os.path.exists(ip):
                print(f"[faiss-sq8] {dim} 无源索引，跳过", flush=True)
                continue
            idx = faiss.read_index(ip)
            n = idx.ntotal
            vecs = idx.reconstruct_n(0, n)          # [n, d] fp32（从 fp32 索引取回全部向量）
            sq = faiss.IndexScalarQuantizer(
                idx.d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
            sq.train(vecs)          # SQ8 需先 train 量化范围，再 add
            sq.add(vecs)
            faiss.write_index(sq, os.path.join(out_dir, f"{dim}.index"))
            shutil.copy(os.path.join(src_dir, f"{dim}.pkl"),
                        os.path.join(out_dir, f"{dim}.pkl"))
            print(f"[faiss-sq8] {dim}: {n} 行 -> {out_dir}/{dim}.index", flush=True)
            del vecs

    # ------------------------------------------------------------------
    # 检索：集合-集合 MaxSim 晚交互
    # ------------------------------------------------------------------
    def search_maxsim(self, query_patch_set, dim: str, topk: int = AGG_POOL_TOPK,
                      ann_topk: int = ANN_TOPK,
                      score_threshold=MAXSIM_SCORE_THRESHOLD,
                      min_coverage: float = MAXSIM_MIN_COVERAGE) -> list:
        idx, meta = self._load(dim)
        q = query_patch_set.float().cpu().numpy()                  # [Nq, D] bf16->fp32
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.clip(norms, 1e-8, None)                         # L2 归一化保险
        Dm, Im = idx.search(q.astype(np.float32), ann_topk)        # [Nq, ann_topk]

        n_q = q.shape[0]
        info = {}
        best_in_set = defaultdict(list)
        for qi in range(n_q):
            per_sample = defaultdict(lambda: float("-inf"))
            for k in range(ann_topk):
                row = int(Im[qi, k])
                if row < 0:
                    continue
                sid, orig, control, text = meta[row]
                sc = float(Dm[qi, k])
                per_sample[sid] = max(per_sample[sid], sc)
                info[sid] = {"sample_id": sid, "orig_image_path": orig,
                             "control_image_path": control, "dim_original_text": text}
            for sid, sc in per_sample.items():
                best_in_set[sid].append(sc)

        if not best_in_set:
            return []

        agg = {sid: sum(vals) / len(vals) for sid, vals in best_in_set.items()}

        if min_coverage and min_coverage > 0:
            agg = {sid: sc for sid, sc in agg.items()
                   if len(best_in_set[sid]) / n_q >= min_coverage}
        if score_threshold is not None:
            agg = {sid: sc for sid, sc in agg.items() if sc >= score_threshold}
        if not agg:
            return []

        ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:topk]
        return [dict(info[sid], score=round(sc, 4)) for sid, sc in ranked]

    # ------------------------------------------------------------------
    def _load(self, dim: str):
        if dim in self._indices:
            return self._indices[dim], self._metas[dim]
        ip = os.path.join(self.index_dir, f"{dim}.index")
        mp = os.path.join(self.index_dir, f"{dim}.pkl")
        if not os.path.exists(ip):
            raise FileNotFoundError(
                f"FAISS 索引不存在: {ip}\n请先运行: python main.py faiss-build")
        idx = faiss.read_index(ip)
        with open(mp, "rb") as f:
            meta = pickle.load(f)
        self._indices[dim] = idx
        self._metas[dim] = meta
        return idx, meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p2 = sub.add_parser("merge", help="合并分段重建的 faiss 索引")
    p2.add_argument("--out", default=FAISS_INDEX_DIR)
    p2.add_argument("--parts", type=int, default=4)
    a = ap.parse_args()
    FaissIndex.merge_parts(a.out, n_parts=a.parts)
