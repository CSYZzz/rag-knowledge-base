"""
重建索引 —— 换 embedding 后端之后必须做，没有例外。

为什么必须重建（面试会问）：

  向量维度变了。bge-small-zh-v1.5 是 512 维，bge-m3 是 1024 维。
  新旧向量混在一个库里，numpy 会直接报 shape 不匹配；
  但如果换成同维度模型，它**不报错**，只是检索结果莫名其妙地变差。
  后者难查得多，因为它看起来"能跑"。

  所以规则是：换 embedding 模型 = 重建索引 + 重跑评测，两件事一起做。

用法：
    set EMBED_BACKEND=remote
    set SILICONFLOW_API_KEY=sk-xxx
    python rebuild_index.py 测试文档.txt
    python eval.py --compare        # 重建后必须重跑，简历上的数字要跟着变
"""

import os
import sys

import minimal_rag as rag


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "测试文档.txt"
    if not os.path.exists(src):
        print(f"找不到文件：{src}")
        return 1

    store = rag.get_store()
    old_chunks = store.count()
    old_dim = store.vectors.shape[1] if old_chunks else None
    print(f"旧索引：{old_chunks} 块，维度 {old_dim}")

    store.clear()
    rag._bm25 = None          # BM25 缓存了旧分块，必须一起清
    rag._reranker = None

    print(f"正在用「{rag.embedder_name() if rag.get_embedder() else '未知'}」重建……")
    rag.index_file(src)

    new_chunks = store.count()
    new_dim = store.vectors.shape[1] if new_chunks else None
    print(f"新索引：{new_chunks} 块，维度 {new_dim}，后端 {rag.embedder_name()}")

    if old_dim and new_dim and old_dim != new_dim:
        print(f"\n[注意] 维度 {old_dim} → {new_dim}，之前的评测结果已作废。")
        print("       请重跑 eval.py，并把简历上的数字更新为新后端的实测值。")
    if rag._degraded:
        print("\n[警告] 当前处于哈希降级状态，这个索引不能用于对外演示。")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
