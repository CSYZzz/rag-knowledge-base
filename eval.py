#!/usr/bin/env python
"""
评测脚本（阶段 3）—— 这个文件决定你简历上那个数字真不真。

为什么这么设计：
  只看"回答对不对"是不够的（阶段 0.5 已经证明：答案对，检索可能是错的）。
  所以这里的指标全部基于「检索到的块里到底有没有答案」，
  用 questions.json 里预先标注的 keywords 判断，不依赖 LLM 生成。
  好处：快、免费、可重复，能一次跑多组配置做对照。

  判断规则：
    某道题的 keywords 出现在第 1 名的块里   → Top1 命中
    出现在 2~4 名的块里                    → 只是召回，不算命中
    一条都没出现                            → 未召回（真的找不到）

用法：
    python eval.py                  跑当前配置
    python eval.py --compare        三种配置对比（不调 API，很快）
    python eval.py --compare -v     同上，并打印失败明细
    python eval.py --rerank         当前配置，但加上 LLM 重排（会调 50 次 API）
"""

import argparse
import json
import os
import sys

import minimal_rag as m

SOURCE = "测试文档.txt"
QUESTION_FILE = "questions.json"


# ============ 数据 ============

def load_questions(path=QUESTION_FILE):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def contains_answer(text, keywords):
    """块里是否含任一答案关键词。keywords 为空表示这题文档里没有答案。"""
    return any(k in text for k in keywords)


# ============ 配置切换 ============

def set_config(chunk_mode, retrieval_mode, rerank_mode):
    """
    直接改模块级常量，并清掉所有缓存。
    因为 split_chunks / retrieve 读的就是这些全局变量，改完立即生效；
    但 store、bm25、reranker 都是懒加载缓存的，必须清空才会按新配置重建。
    """
    m.CHUNK_MODE = chunk_mode
    m.RETRIEVAL_MODE = retrieval_mode
    m.RERANK_MODE = rerank_mode
    m._store = None
    m._bm25 = None
    m._reranker = None


# ============ 评测 ============

def evaluate(questions, verbose=False):
    qs = [q for q in questions if q["type"] != "unanswerable"]
    top1 = recall = 0
    ranks = []
    fails = []

    for q in qs:
        hits = m.retrieve(q["q"])
        texts = [t for t, _meta, _score in hits]

        rank = None
        for i, t in enumerate(texts, 1):
            if contains_answer(t, q["keywords"]):
                rank = i
                break

        if rank == 1:
            top1 += 1
        if rank:
            recall += 1
            ranks.append(rank)
        else:
            ranks.append(99)          # 未召回的惩罚值，只用于算平均名次

        if rank != 1:
            first = texts[0].split("\n")[0] if texts else "（空）"
            fails.append((q["id"], q["q"], rank, first))

    return {
        "n": len(qs),
        "top1": top1,
        "recall": recall,
        "avg_rank": (sum(ranks) / len(ranks)) if ranks else 0.0,
        "fails": fails,
        "ranks": ranks,
    }


def print_result(name, st, verbose=False):
    n = st["n"]
    p1 = st["top1"] / n * 100 if n else 0
    pr = st["recall"] / n * 100 if n else 0
    avg = st["avg_rank"]
    print(f"{name:<32} {st['top1']:>3}/{n:<3} {p1:>5.1f}%   "
          f"{st['recall']:>3}/{n:<3} {pr:>5.1f}%   {avg:>6.2f}")

    if verbose and st["fails"]:
        for qid, q, rank, first in st["fails"]:
            mark = f"第{rank}名" if rank else "未召回"
            print(f"      #{qid:<3} {q[:26]:<28} {mark:<8} 实际Top1→ {first}")


# ============ 主流程 ============

def run_once(name, chunk, retrieval, rerank, questions, verbose):
    set_config(chunk, retrieval, rerank)
    m.index_file(SOURCE)
    st = evaluate(questions, verbose)
    print_result(name, st, verbose)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="三种配置对比")
    ap.add_argument("--rerank", action="store_true",
                    help="启用 LLM 重排（会调用 API，慢且花钱）")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印失败明细")
    args = ap.parse_args()

    questions = load_questions()
    total = len(questions)
    ans_cnt = sum(1 for q in questions if q["type"] == "unanswerable")
    print(f"题库：{total} 题（可评测 {total - ans_cnt} 题，不可答 {ans_cnt} 题）")
    print(f"文档：{SOURCE}")
    print(f"Embedding：{m.embedder_name()}")
    print()
    print(f"{'配置':<32} {'Top1命中':<11} {'Recall@4':<12} {'平均名次'}")
    print("-" * 82)

    if args.compare:
        run_once("朴素分块 + 纯向量", "naive", "vector", "none", questions, args.verbose)
        run_once("智能分块 + 纯向量", "smart", "vector", "none", questions, args.verbose)
        run_once("智能分块 + 混合检索", "smart", "hybrid", "none", questions, args.verbose)
        if args.rerank:
            run_once("智能分块 + 混合检索 + 重排", "smart", "hybrid", "llm",
                     questions, args.verbose)
    else:
        rr = "llm" if args.rerank else "none"
        run_once(f"当前配置（分块={m.CHUNK_MODE} 检索={m.RETRIEVAL_MODE} 重排={rr}）",
                 m.CHUNK_MODE, m.RETRIEVAL_MODE, rr, questions, args.verbose)

    print("-" * 82)
    print("平均名次：命中块排在第几名，越接近 1 越好；未召回按 99 计入。")

    # 降级时出的数字不能写进简历，也不能和别人比。这里必须显式拦住。
    if m._degraded:
        print()
        print("!! 警告：BGE 模型没加载成功，本次结果是【哈希降级】跑出来的，")
        print("!! 和 README 里的基线不可比，不要引用这些数字。")
        print("!! 排查：pip install -U sentence-transformers --upgrade")


if __name__ == "__main__":
    main()
