# -*- coding: utf-8 -*-
"""
常驻 HTTP 推理服务：4B 编码器 + faiss 索引常驻内存，单张查询返回动态份数方案文本
（每份 = 每维第 k 张候选好图 + 文本；方案数 = 各维候选好图数的最小值，图编后台异步）。

启动：
  python server.py --port 8100 --n-solutions 5
调用（JSON base64 图片）：
  curl -X POST localhost:8100/infer -H 'Content-Type: application/json' \\
       -d '{"image_base64": "<base64 of jpg>", "stem": "photo1"}'
返回：
  {"output_dir", "solutions":[{"index","direction","text"},...],
   "evidence_by_dim", "retrieval_s", "edit_status":"processing", "total_s"}
健康检查：
  curl localhost:8100/health
"""
from __future__ import annotations

import os
# 必须在任何 BLAS/torch/faiss 初始化前设置：避免 faiss(OpenMP) 与 torch/HTTP 线程混用触发
# OpenBLAS "too many memory regions" 崩溃
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import base64
import io
import json
import multiprocessing as mp
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DEVICE, DIM_ORDER, DTYPE, ENCODER_MODEL_PATH,  # noqa: E402
                    GENERATE_MAX_NEW_TOKENS, N_SOLUTIONS, OUTPUT_DIR)
from encoder import DdgeEncoder                                   # noqa: E402
from faiss_index import FaissIndex                                # noqa: E402
from aesrecon import build_variant_map                            # noqa: E402
from user import retrieve_dim_evidence                            # noqa: E402
from prompt import GuidanceGenerator                              # noqa: E402


class InferenceEngine:
    """常驻引擎：模型 / 索引 / 变体映射只加载一次，各请求复用。"""

    def __init__(self, output_dir: str = OUTPUT_DIR,
                 n_solutions: int = N_SOLUTIONS,
                 max_new_tokens: int = GENERATE_MAX_NEW_TOKENS):
        print("[server] 加载 4B 编码器 ...", flush=True)
        self.encoder = DdgeEncoder(model_path=ENCODER_MODEL_PATH,
                                   dtype=DTYPE, device=DEVICE)
        print("[server] 加载 faiss 索引（7 维，一次性预加载）...", flush=True)
        self.client = FaissIndex()
        for d in DIM_ORDER:
            self.client._load(d)
        print("[server] 加载 variant_map ...", flush=True)
        self.variant_map = build_variant_map()
        self.generator = GuidanceGenerator()
        self.output_dir = output_dir
        self.n_solutions = n_solutions
        self.max_new_tokens = max_new_tokens
        cap_s = "不限" if not n_solutions else str(n_solutions)
        print(f"[server] 引擎就绪（方案上限 {cap_s} / {max_new_tokens} tokens）", flush=True)

    def infer(self, image_bytes: bytes, stem: str = "photo") -> dict:
        """图片 -> 检索 + 动态份数方案文本（图编异步后台），返回结果 dict。
        全程加锁串行：faiss 的 OpenMP 线程池/GPU 模型在多线程并发访问下会段错误。"""
        with _INFER_LOCK:
            return self._infer_unlocked(image_bytes, stem)

    def _infer_unlocked(self, image_bytes: bytes, stem: str) -> dict:
        user_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        os.makedirs(self.output_dir, exist_ok=True)
        tmp_path = os.path.join(self.output_dir, f"_tmp_{stem or 'photo'}.jpg")
        user_img.save(tmp_path, "JPEG", quality=95)

        t_enc0 = time.time()
        query_sets = self.encoder.get_dim_patch_sets_parallel(user_img)
        encode_s = time.time() - t_enc0

        t_faiss0 = time.time()
        evidence_by_dim = retrieve_dim_evidence(query_sets, self.client, self.variant_map)
        faiss_search_s = time.time() - t_faiss0
        retr_s = encode_s + faiss_search_s

        solutions, js_path, _ = self.generator.generate_solutions_and_save(
            tmp_path, evidence_by_dim, output_dir=self.output_dir,
            n_solutions=self.n_solutions, max_new_tokens=self.max_new_tokens,
            retrieval_s=retr_s, encode_s=encode_s, faiss_search_s=faiss_search_s,
            image_edit_async=True,
        )
        return {
            "output_dir": os.path.dirname(js_path),
            "solutions": solutions,
            "evidence_by_dim": evidence_by_dim,
            "retrieval_s": round(retr_s, 2),
            "edit_status": "processing",
        }


_INFER_LOCK = threading.Lock()


# 推理 worker 进程：独立进程加载引擎（模型+faiss 索引常驻），隔离 CUDA/BLAS 环境。
# HTTP 主进程不加载引擎（不初始化 CUDA），请求通过队列转发给 worker。
_REQ_Q: mp.Queue | None = None
_RESP_Q: mp.Queue | None = None


def worker_main(req_q: mp.Queue, resp_q: mp.Queue, output: str,
                n_solutions: int, max_new_tokens: int) -> None:
    """worker 进程：加载引擎，循环处理请求。"""
    engine = InferenceEngine(output_dir=output, n_solutions=n_solutions,
                             max_new_tokens=max_new_tokens)
    print("[server] worker 引擎就绪", flush=True)
    while True:
        rid, image_bytes, stem = req_q.get()
        try:
            result = engine._infer_unlocked(image_bytes, stem)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            result = {"error": str(e)}
        resp_q.put((rid, result))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静默默认访问日志
        pass

    def _json(self, code: int, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/infer":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            img_b64 = body.get("image_base64") or body.get("image")
            stem = str(body.get("stem", "photo"))
            if not img_b64:
                return self._json(400, {"error": "missing image_base64"})
            if "," in img_b64:                      # data:image/...;base64, 前缀剥离
                img_b64 = img_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(img_b64)
            t_all0 = time.time()
            rid = id(self)                       # 请求唯一标识
            _REQ_Q.put((rid, image_bytes, stem))
            while True:                          # 等待 worker 返回本请求结果
                r_rid, result = _RESP_Q.get()
                if r_rid == rid:
                    break
            if "error" in result:
                return self._json(500, result)
            result["total_s"] = round(time.time() - t_all0, 2)
            return self._json(200, result)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return self._json(500, {"error": str(e)})


def run_server(port: int = 8100, host: str = "0.0.0.0",
               n_solutions: int = N_SOLUTIONS,
               max_new_tokens: int = GENERATE_MAX_NEW_TOKENS,
               output: str = OUTPUT_DIR) -> None:
    global _REQ_Q, _RESP_Q
    # 用 spawn：PyTorch 禁止在 fork 子进程里初始化 CUDA；spawn 全新进程重新 import + 首次 CUDA 合法
    ctx = mp.get_context("spawn")
    _REQ_Q = ctx.Queue(maxsize=8)
    _RESP_Q = ctx.Queue(maxsize=8)
    wp = ctx.Process(target=worker_main, daemon=True,
                     args=(_REQ_Q, _RESP_Q, output, n_solutions, max_new_tokens))
    wp.start()
    print(f"[server] worker 已启动（spawn），监听 http://{host}:{port} （Ctrl+C 停止）", flush=True)
    HTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description="常驻 DDGE 推理服务")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--n-solutions", type=int, default=N_SOLUTIONS)
    ap.add_argument("--max-new-tokens", type=int, default=GENERATE_MAX_NEW_TOKENS)
    ap.add_argument("--output", type=str, default=OUTPUT_DIR)
    a = ap.parse_args()
    run_server(port=a.port, host=a.host, n_solutions=a.n_solutions,
               max_new_tokens=a.max_new_tokens, output=a.output)


if __name__ == "__main__":
    main()
