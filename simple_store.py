"""
极简向量库 + 兜底 embedding —— 用来替代 chromadb 和 sentence-transformers。

为什么要自己写：
1) chromadb 在 Python 3.13 上装不上，一个包卡住整个项目不值得；
2) 数据量小（几百到几千块）时，暴力算余弦相似度比建索引还快；
3) 代码只有一百多行，面试时能讲清每一行在干什么——这比"我调了 chromadb 的 API"
   更能证明你懂原理。这句话本身就是简历上的一个亮点。

它做了什么：
- 存：文本 + 向量 + 元数据（来源文件名、第几块）
- 查：给一个查询向量，算余弦相似度，返回最像的 k 条
- 持久化：向量存 .npy，文本存 .json，重启不丢
"""

import json
import zlib
from pathlib import Path

import numpy as np


# ============================================================
# 一、兜底 embedding：字符 2-gram 哈希
# ============================================================

def _char_ngrams(text: str, n: int = 2):
    """把一段中文切成连续 n 个字的小片段。不依赖分词库，中文够用。"""
    text = "".join(ch for ch in text if not ch.isspace())
    if len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


class HashEmbedder:
    """
    不需要下载任何模型的 embedding 方案。

    做法：把文字切成 2 字片段 → 每个片段用 CRC32 映射到 0~2047 之间的一个位置 →
    该位置加 1 → 最后做 L2 归一化。这叫 hashing vectorizer。

    效果肯定不如 BGE 模型，但完全离线、零依赖、启动一秒。
    用它兜底的真正价值是：让整条管道先跑通，别卡在环境上。
    """

    needs_prefix = False   # 只有 BGE 模型需要加 query 前缀

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def encode(self, texts, **kwargs):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for gram in _char_ngrams(t):
                # 用 crc32 而不是内置 hash()：内置 hash 每次启动的种子不同，
                # 会导致重启后向量对不上，检索全乱。
                out[i, zlib.crc32(gram.encode("utf-8")) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)

    def __repr__(self):
        return f"HashEmbedder(dim={self.dim})"


# ============================================================
# 二、向量库
# ============================================================

class SimpleStore:
    """
    文件级向量库。

    存储结构：
        vector_store/vectors.npy   —— 二维数组，每行是一个块的向量
        vector_store/meta.json     —— 每行对应的文本、来源文件、块序号

    检索原理：
        向量都做过 L2 归一化，所以 点积 = 余弦相似度。
        给一个查询向量 q，算 vectors @ q，越大越像，取前 k 个。
    """

    def __init__(self, path: str = "./vector_store"):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vec_file = self.dir / "vectors.npy"
        self.meta_file = self.dir / "meta.json"

        self.vectors = None      # shape (N, D)
        self.meta = []           # list[dict]，与 vectors 行一一对应
        self._load()

    # ---------- 持久化 ----------

    def _load(self):
        if self.vec_file.exists() and self.meta_file.exists():
            self.vectors = np.load(self.vec_file).astype(np.float32)
            self.meta = json.loads(self.meta_file.read_text(encoding="utf-8"))

    def save(self):
        if self.vectors is None:
            return
        np.save(self.vec_file, self.vectors)
        self.meta_file.write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    # ---------- 写入 ----------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(norms, 1e-9, None)

    def delete_source(self, source: str):
        """同一份文档重新索引前，先删掉旧的，避免重复。"""
        if not self.meta:
            return
        keep = [i for i, m in enumerate(self.meta) if m["source"] != source]
        if len(keep) == len(self.meta):
            return
        self.vectors = self.vectors[keep]
        self.meta = [self.meta[i] for i in keep]
        self.save()

    def add(self, texts, embeddings, source: str):
        v = np.asarray(embeddings, dtype=np.float32)
        v = self._normalize(v)

        # 换了 embedding 模型会导致维度变化（哈希 2048 维 / BGE 512 维），
        # 新旧混在一起没法算相似度，所以直接清空重建，并明确告诉你为什么。
        if self.vectors is not None and self.vectors.shape[1] != v.shape[1]:
            print(f"[提示] 向量维度 {self.vectors.shape[1]} → {v.shape[1]}"
                  f"（embedding 方式变了），自动清空旧库重建。")
            self.clear()

        self.vectors = v if self.vectors is None else np.vstack([self.vectors, v])
        # 块号从 1 开始，和用户看到的"第 N 块"一致，避免界面上出现"第 0 块"
        for i, t in enumerate(texts, 1):
            self.meta.append({"source": source, "chunk": i, "text": t})
        self.save()

    # ---------- 检索 ----------

    def query_with_index(self, embedding, top_k: int = 4):
        """
        和 query 一样，但额外返回全局索引（0..N-1）。
        混合检索做 RRF 融合时需要靠它把两路结果对齐。
        """
        if self.vectors is None or not self.meta:
            return []

        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-9)

        sims = self.vectors @ q                      # 余弦相似度
        top_k = min(top_k, len(sims))
        idx = np.argsort(-sims)[:top_k]              # 从大到小

        return [
            (int(i), self.meta[i]["text"], self.meta[i], float(1.0 - sims[i]))
            for i in idx
        ]

    def query(self, embedding, top_k: int = 4):
        """返回 [(文本, 元数据, 距离), ...]，距离越小越相似（1 - 余弦相似度）。"""
        return [
            (text, meta, dist)
            for _, text, meta, dist in self.query_with_index(embedding, top_k)
        ]

    def count(self) -> int:
        return len(self.meta)

    def clear(self):
        self.vectors = None
        self.meta = []
        if self.vec_file.exists():
            self.vec_file.unlink()
        if self.meta_file.exists():
            self.meta_file.unlink()
