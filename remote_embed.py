"""
远程 embedding 后端 —— 把向量化从「本地模型」换成「API 调用」。

为什么需要这一层（面试必答）：

  免费部署层（Render Free）只有 512MB 内存、0.1 核 CPU。
  而 sentence-transformers + torch 启动一个 BGE 小模型，常驻就要 500MB 以上，
  容器必然 OOM 被杀。镜像体积也会从 200MB 涨到 2GB，构建时间从 1 分钟变 10 分钟。

  所以线上把向量化外包出去：硅基流动的 BAAI/bge-m3 免费、不限量、大陆直连。
  换完之后常驻内存约 120MB，镜像约 200MB，512MB 装得下。

  代价（这些必须能讲清楚，否则面试官会认为你只是"抄了个 API"）：
    1. 每次查询多一次网络往返，实测约 +80~150ms
    2. 多了一个外部依赖，它挂了服务就残废 → 所以保留本地模型和哈希两条后路
    3. 向量维度变了：bge-small-zh-v1.5 是 512 维，bge-m3 是 1024 维，
       **已有索引必须重建，新旧向量绝对不能混用**。
       混用的表现不是报错，而是检索结果莫名其妙地差——这是最阴的一类 bug。

为什么不直接把索引也放到远端：
  索引只有几十 KB，打进镜像最省事也最稳。放远端要处理版本一致性、拉取失败、
  冷启动等待，收益为零。
"""

from __future__ import annotations

import os
import time

import numpy as np

DEFAULT_BASE = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "BAAI/bge-m3"

# 单批上限。bge-m3 本身支持更大批量，但免费层有速率限制，
# 批量太大会撞 429，反而更慢。32 是实测下来比较稳的值。
BATCH_SIZE = 32

# 重试次数。免费层偶发 429/超时，重试一次基本能过去；
# 重试太多会把一次慢查询拖成一次超时查询。
MAX_RETRIES = 2


class RemoteEmbedder:
    """接口与 SentenceTransformer.encode 保持一致，上层代码不用改。"""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None):
        key = api_key or os.environ.get("SILICONFLOW_API_KEY", "")
        if not key:
            raise RuntimeError("缺少 SILICONFLOW_API_KEY，无法启用远程 embedding")

        from openai import OpenAI

        self._client = OpenAI(
            api_key=key,
            base_url=base_url or os.environ.get("SILICONFLOW_BASE_URL", DEFAULT_BASE),
            timeout=20.0,
        )
        self.model = model or os.environ.get("EMBED_MODEL_REMOTE", DEFAULT_MODEL)
        self.dim: int | None = None

        # 累计计数，监控面板要用。embedding 调用次数直接对应外部 API 消耗。
        self.calls = 0
        self.texts_done = 0
        self.failures = 0

    # ---------- 对外接口 ----------

    def encode(self, texts, **kwargs) -> np.ndarray:
        """输入 str 或 list[str]，返回 float32 的 (n, dim) 数组。"""
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)

        if not texts:
            return np.zeros((0, self.dim or 1024), dtype=np.float32)

        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[i:i + BATCH_SIZE]))

        arr = np.asarray(vectors, dtype=np.float32)
        self.dim = arr.shape[1]
        self.texts_done += len(texts)
        return arr

    def get_sentence_embedding_dimension(self) -> int | None:
        """和 SentenceTransformer 同名，避免上层调用处要做兼容分支。"""
        return self.dim

    def backend_name(self) -> str:
        return f"远程 {self.model}"

    # ---------- 内部实现 ----------

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_err = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float",
                )
                self.calls += 1

                # API 不保证返回顺序与输入一致，按 index 排回去。
                # 不排序的话，向量会和文本错位——检索不报错，但答案会张冠李戴。
                items = sorted(resp.data, key=lambda d: d.index)
                if len(items) != len(batch):
                    raise RuntimeError(
                        f"返回条数不符：请求 {len(batch)} 条，返回 {len(items)} 条"
                    )
                return [d.embedding for d in items]

            except Exception as e:
                last_err = e
                self.failures += 1
                if attempt < MAX_RETRIES:
                    # 线性退避：免费层撞限流时，等一下就能过；
                    # 指数退避在这里反而会让用户等太久。
                    time.sleep(0.8 * (attempt + 1))

        raise RuntimeError(f"远程 embedding 调用失败（已重试 {MAX_RETRIES} 次）：{last_err}")


def probe() -> tuple[bool, str]:
    """
    启动自检：上线前调一次，确认 key 有效、模型可用、维度符合预期。

    这一步的价值在于把「配置错误」从"用户第一次提问时才暴露"
    提前到"服务启动时就能看见"，排查成本从半小时变成三秒。
    """
    try:
        emb = RemoteEmbedder()
        vec = emb.encode(["自检"])
        return True, f"{emb.model} 可用，维度 {vec.shape[1]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    ok, msg = probe()
    print(("[OK] " if ok else "[FAIL] ") + msg)
