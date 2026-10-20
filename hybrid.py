"""
混合检索 + 重排（阶段 2）

为什么需要这三段，理由全部来自阶段 0.6 和阶段 1 的实测数据：

  - 纯向量检索：Recall@4 = 100%，但 Top1 排序在临界点抖动
    （「设备能装在室外吗」第1名 0.5857 vs 第2名 0.5980，只差 0.0123；
      「通信失败怎么排查」第1名 vs 第2名只差 0.0001）
    结论：不是找不到，是排错了。换个更好的 embedding 治不了这个。

  - Top1 错的那次，错因是「设备」撞上了「设备地址」这种高频词干扰。
    纯向量的语义打分对这种干扰不敏感，而关键词检索（BM25）恰恰擅长压住它。

  - 所以：一路关键词、一路向量，融合后再精排。
    这是一个标准的 two-stage retrieval 架构。

三段各自的职责：

  1. BM25        —— 关键词召回，补上语义模型容易平均掉的精确词
  2. RRF 融合     —— 两路分数尺度不同没法直接加权，用"排名倒数"融合，免调权重
  3. Reranker    —— 对粗筛出的少量候选做精排（默认用 LLM，见类注释）
"""

import math
import os
import re

# 中文按 2 字滑窗切，英文/数字整体保留。
# 用 2-gram 而不是分词库，是为了不额外引入 jieba 这类依赖，
# 中文短查询在这个粒度上已经够用。
_TOKEN_RE = re.compile(r"[\u4e00-\u9fa5]+|[a-zA-Z][a-zA-Z0-9_.\-]*|\d+")


def tokenize(text: str):
    tokens = []
    for m in _TOKEN_RE.finditer(text.lower()):
        s = m.group()
        if "\u4e00" <= s[0] <= "\u9fa5":
            if len(s) == 1:
                tokens.append(s)
            else:
                tokens.extend(s[i:i + 2] for i in range(len(s) - 1))
        else:
            tokens.append(s)
    return tokens


class BM25:
    """
    Okapi BM25。k1 控制词频饱和速度，b 控制长度归一化强度。
    b=0.75 是经验默认值：块长差异大时尤其重要。
    """

    def __init__(self, docs, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.tf = []          # 每篇文档的 {term: 词频}
        self.doc_len = []
        self.df = {}          # term -> 出现在几篇文档里

        for d in docs:
            toks = tokenize(d)
            self.doc_len.append(len(toks))
            counter = {}
            for t in toks:
                counter[t] = counter.get(t, 0) + 1
            self.tf.append(counter)
            for t in counter:
                self.df[t] = self.df.get(t, 0) + 1

        self.avg_len = (sum(self.doc_len) / self.N) if self.N else 0.0

    def scores(self, query: str):
        """返回一个长度 N 的分数列表，与 docs 顺序对齐。"""
        q_tokens = tokenize(query)
        out = [0.0] * self.N
        if not self.N or self.avg_len <= 0:
            return out

        for t in q_tokens:
            df = self.df.get(t)
            if not df:
                continue
            # 加 1 保证 IDF 非负：稀有词权重高，常见词接近 0
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for i in range(self.N):
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out

    def rank(self, query: str, top_k: int = None):
        """按 BM25 分数从高到低返回文档索引。"""
        sc = self.scores(query)
        order = sorted(range(self.N), key=lambda i: -sc[i])
        if top_k is not None:
            order = order[:top_k]
        return order


def rrf_fuse(rankings, k: int = 60, weights=None):
    """
    Reciprocal Rank Fusion。

        score(d) = Σ  1 / (k + rank_i(d))

    为什么不用加权求和：
    向量的余弦距离和 BM25 的分数不在一个尺度上（前者 0~2，后者可能到十几），
    加权求和的聚合方式本身是线性的，而分数的绝对值没有可比性。
    RRF 只看排名、不看分数，天然跨尺度，k=60 是原论文的经验值。

    rankings: 每个检索器的结果，元素是文档索引，已按相关性降序
    返回：[(文档索引, 融合分数), ...] 降序
    """
    fused = {}
    for ri, ranking in enumerate(rankings):
        w = weights[ri] if weights else 1.0
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class Reranker:
    """
    重排器，三种后端：

      llm   —— 用大模型给候选排序（默认）。不需要下载任何模型，代价是每次检索
                多一次 API 调用、多 1~3 秒。对中小知识库完全够用。
      cross —— 用本地 CrossEncoder 小模型。推理快、不花钱，但 bge-reranker-base
                基于 xlm-roberta，权重 1.1GB，国内网络实测拉不下来，所以默认不用。
      none  —— 不重排，直接用 RRF 融合顺序。

    为什么重排有用（本来该用 cross-encoder 的理由）：
      - bi-encoder（检索用的 BGE）：问题和文档分别编码成向量再算相似度。
        文档可以预先编码、检索快，但问题和文档"从没见过面"。
      - cross-encoder：把 (问题, 文档) 拼成一对一起送进模型，能直接看到词与词的
        关系，精度高得多；但每个候选都要跑一次前向，慢，所以只能用在少量候选上。
      这就是「两阶段检索」的由来：先粗筛 10 条，再精排 4 条。

    LLM 重排是同一件事的另一种实现：把"成对判断相关性"交给通用大模型，
    省掉下载和部署，代价是延迟和 token 成本。这是工程取舍，面试可以讲。
    """

    def __init__(self, mode: str = "llm", model_name: str = "BAAI/bge-reranker-base"):
        self.mode = (mode or "none").lower()
        self.model_name = model_name
        self.model = None

    # ---------- 对外接口 ----------

    def rerank(self, query: str, candidates, top_k: int = 4):
        """
        candidates: [(全局索引, 文本), ...]
        返回 [(全局索引, 文本, 分数), ...]，分数越大越相关。
        任何失败都退回原顺序，绝不向上抛异常——检索不能因为重排挂了就不工作。
        """
        if not candidates:
            return []

        if self.mode == "llm":
            out = self._llm_rerank(query, candidates, top_k)
            return out if out is not None else self._passthrough(candidates)

        if self.mode == "cross":
            out = self._cross_rerank(query, candidates)
            return out if out is not None else self._passthrough(candidates)

        return self._passthrough(candidates)

    @staticmethod
    def _passthrough(candidates):
        return [(i, t, 0.0) for i, t in candidates]

    # ---------- LLM 重排 ----------

    def _llm_rerank(self, query, candidates, top_k):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("[提示] 没有 DEEPSEEK_API_KEY，本次跳过重排。")
            return None

        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        n = len(candidates)
        # 注意：这里必须给候选块全文，不能截断。
        # 之前截到前 100 字，导致「现象四：继电器频繁动作」「120 欧姆终端电阻」
        # 这类位于块中后部的答案对重排模型不可见，Recall@4 从 97.8% 掉到 91.3%。
        listing = "\n\n".join(
            f"[{i}] {text}"
            for i, (_idx, text) in enumerate(candidates)
        )
        prompt = (
            "下面是从一份文档里检索到的资料片段。请按与问题的相关度给它们排序。\n\n"
            f"问题：{query}\n\n"
            f"片段：\n{listing}\n\n"
            f"请输出全部 {n} 个编号，按相关度从高到低，用英文逗号分隔，"
            "例如：3,1,5,0,2,4,6,7,8,9\n"
            "必须包含每个编号且不重复。不要输出任何解释文字。"
        )

        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=64,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[提示] LLM 重排出错：{type(e).__name__}，本次退回融合顺序。")
            return None

        order = []
        for n in re.findall(r"\d+", text):
            i = int(n)
            if 0 <= i < len(candidates) and i not in order:
                order.append(i)
        if not order:
            print(f"[提示] LLM 返回无法解析（{text[:30]!r}），退回融合顺序。")
            return None
        for i in range(len(candidates)):
            if i not in order:
                order.append(i)

        # 用排名倒数当分数：1.0 第一，0.5 第二，越大越相关
        return [
            (candidates[i][0], candidates[i][1], 1.0 / (rank + 1))
            for rank, i in enumerate(order)
        ]

    # ---------- 本地 CrossEncoder 重排 ----------

    def _cross_rerank(self, query, candidates):
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                print(f"正在加载重排模型 {self.model_name} ...")
                self.model = CrossEncoder(self.model_name)
                print("重排模型加载完成。\n")
            except Exception as e:
                print(f"[提示] 重排模型加载失败：{type(e).__name__}，退回融合顺序。")
                return None

        try:
            scores = self.model.predict([(query, t) for _i, t in candidates])
        except Exception as e:
            print(f"[提示] 重排出错：{type(e).__name__}，退回融合顺序。")
            return None

        out = [(idx, text, float(s)) for (idx, text), s in zip(candidates, scores)]
        out.sort(key=lambda x: -x[2])
        return out
