"""
最小可运行的 RAG —— 第 0 阶段
目标：一个文档进去，一个问题出来，拿到带出处的回答。

已改用自写的 SimpleStore（simple_store.py）替代 chromadb，
因为 chromadb 在 Python 3.13 上装不上。功能完全一样。

用法：
    1) pip install -r requirements.txt
    2) set DEEPSEEK_API_KEY=sk-xxx
    3) python minimal_rag.py load 测试文档.txt
    4) python minimal_rag.py 保修期是多久

第一次运行会尝试加载 bge-small-zh 模型（约 95MB）；
加载不到会自动降级为字符 2-gram 哈希检索，流程不受影响。
"""

import os
import re
import sys
from pathlib import Path

from simple_store import HashEmbedder, SimpleStore
from hybrid import BM25, Reranker, rrf_fuse

# ============ 配置 ============

LLM_MODEL     = "deepseek-chat"
EMBED_MODEL   = "BAAI/bge-small-zh-v1.5"
STORE_DIR     = "./vector_store"
CHUNK_SIZE    = 400        # 每块字数上限
CHUNK_MIN     = 100        # 小于这个字数的独立段，合并到下一块
CHUNK_OVERLAP = 80         # 只在单段超长、退回滑窗时才用
TOP_K         = 4          # 检索返回几块

# 分块策略：smart = 按标题+段落切（阶段 1）；naive = 旧的滑动窗口（用于对比实验）
# 切换方式：set CHUNK_MODE=naive
CHUNK_MODE    = os.environ.get("CHUNK_MODE", "smart")

# 检索策略：hybrid = 向量 + BM25 + RRF + 重排（阶段 2）；vector = 只用向量（阶段 1 的行为）
# 切换方式：set RETRIEVAL_MODE=vector
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid")
CANDIDATE_K    = 10                                   # 粗筛候选数，再由重排压到 TOP_K
# 重排后端：llm = 用大模型排（默认，不用下载）；cross = 本地 CrossEncoder；none = 不重排
RERANK_MODE    = os.environ.get("RERANK_MODE", "llm")

# BGE 模型检索时，query 需要加这个前缀，效果明显更好（官方要求）
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


# ============ 1. 文档解析 ============

def load_text(path: str) -> str:
    """支持 txt / md / pdf，其他格式后面再加"""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(f"[第{i + 1}页]\n{text}")
        return "\n".join(pages)

    raise ValueError(f"暂不支持的格式：{suffix}")


# ============ 2. 分块 ============

# 标题识别：Markdown 的 # / 中文序号「一、二、」/ 「第一章」
_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s+\S"
    r"|[一二三四五六七八九十]+\s*[、.．]\s*\S"
    r"|第[一二三四五六七八九十百]+[章节篇部]\s*\S)"
)


def _is_heading(line: str) -> bool:
    return bool(_HEADING_RE.match(line))


def _clean_title(line: str) -> str:
    return line.strip().lstrip("#").strip()


def parse_sections(text: str, min_size: int = CHUNK_MIN):
    """
    按标题切分成 [(标题, 正文), ...]，保证一块不会跨章节。

    文档最前面如果有一段没有标题的短前言（比如书名），合并到第一节里，
    避免产生一个只有九个字的碎块。
    """
    sections, cur_title, cur_lines = [], "", []

    def flush():
        nonlocal cur_lines
        body = "\n".join(cur_lines).strip()
        if body:
            sections.append((cur_title, body))
        cur_lines = []

    for line in text.split("\n"):
        if _is_heading(line):
            flush()
            cur_title = _clean_title(line)
        else:
            cur_lines.append(line)
    flush()

    # 合并没有标题的短前言
    merged, carry = [], ""
    for title, body in sections:
        if not title and len(body) < min_size:
            carry = body + "\n"
            continue
        merged.append((title, (carry + body).strip()))
        carry = ""
    if carry:
        if merged:
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n" + carry.strip())
        else:
            merged.append(("", carry.strip()))

    return merged


def _split_paragraphs(body: str):
    """先按空行切段；没有空行就按单行切（手册类文档通常一行一个条目）。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in body.split("\n") if p.strip()]
    return paras


def _sliding(text: str, size: int, overlap: int):
    """单段超长时的兜底：滑动窗口。"""
    out, start = [], 0
    step = max(1, size - overlap)
    while start < len(text):
        seg = text[start:start + size].strip()
        if seg:
            out.append(seg)
        start += step
    return out


def smart_split(text: str, max_size: int = CHUNK_SIZE, min_size: int = CHUNK_MIN):
    """
    智能分块（阶段 1 的核心改动）：

    1) 按标题切节，一块绝不跨节
    2) 节内按段落/行打包，尽量装满但不超过 max_size
    3) 只有单段自己就超长，才退回滑动窗口
    4) 每块前面加「【章节名】」，脱离上下文也能读懂（上下文增强）

    和旧方案的区别：旧的是按字数硬切，一句话会被劈成两半；
    现在是先尊重文档结构，再谈长度。
    """
    chunks = []

    for title, body in parse_sections(text, min_size):
        buf, buf_len = [], 0

        for p in _split_paragraphs(body):
            if len(p) > max_size:
                if buf:
                    chunks.append((title, "\n".join(buf)))
                    buf, buf_len = [], 0
                for seg in _sliding(p, max_size, CHUNK_OVERLAP):
                    chunks.append((title, seg))
                continue

            if buf and buf_len + len(p) + 1 > max_size:
                chunks.append((title, "\n".join(buf)))
                buf, buf_len = [], 0

            buf.append(p)
            buf_len += len(p) + 1

        if buf:
            chunks.append((title, "\n".join(buf)))

    out = []
    for title, content in chunks:
        prefix = f"【{title}】\n" if title else ""
        out.append(prefix + content)
    return out


def naive_split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """旧的滑动窗口分块，保留下来做对比实验用。"""
    chunks, start = [], 0
    step = max(1, size - overlap)
    while start < len(text):
        seg = text[start:start + size].strip()
        if seg:
            chunks.append(seg)
        start += step
    return chunks


def split_chunks(text: str):
    """按 CHUNK_MODE 选策略。"""
    if CHUNK_MODE == "naive":
        return naive_split(text)
    return smart_split(text)


# ============ 3. 向量化（带自动降级） ============

_embedder = None
_needs_prefix = False
_degraded = False          # 是否发生了降级，评测脚本要据此拒绝出数


def get_embedder():
    global _embedder, _needs_prefix, _degraded
    if _embedder is not None:
        return _embedder

    try:
        from sentence_transformers import SentenceTransformer
        print(f"正在加载 embedding 模型 {EMBED_MODEL} ...")
        print("（第一次运行需下载约 95MB，慢是正常的）")
        _embedder = SentenceTransformer(EMBED_MODEL)
        _needs_prefix = True
        print("模型加载完成：使用 BGE 语义向量。\n")
    except Exception as e:
        print(f"[提示] 加载 {EMBED_MODEL} 失败：{type(e).__name__}: {e}")
        print("[降级] 改用字符 2-gram 哈希检索，不下载任何模型。")
        print("       效果弱一些，但整条流程完全一样。\n")
        _embedder = HashEmbedder()
        _needs_prefix = False
        _degraded = True

    return _embedder


def embedder_name() -> str:
    """当前实际用的是哪个 embedding，评测报告必须标明，否则数字没有可比性。"""
    get_embedder()
    return "BGE语义向量" if not _degraded else "哈希降级(非BGE)"


def embed(texts, is_query: bool = False):
    model = get_embedder()
    if is_query and _needs_prefix:
        texts = [QUERY_PREFIX + t for t in texts]
    return model.encode(texts)


# ============ 4. 向量库 ============

_store = None


def get_store() -> SimpleStore:
    global _store
    if _store is None:
        _store = SimpleStore(STORE_DIR)
    return _store


def index_file(path: str):
    text = load_text(path)
    chunks = split_chunks(text)
    if not chunks:
        print("文件里没读到内容，检查一下是不是扫描版 PDF")
        return

    store = get_store()
    name = Path(path).name
    store.delete_source(name)          # 同一文档重新索引前先清掉旧的
    store.add(chunks, embed(chunks), name)

    print(f"已索引 {name}：{len(chunks)} 块（模式={CHUNK_MODE}，库中共 {store.count()} 块）")
    lens = [len(c) for c in chunks]
    print(f"块长度分布：最短 {min(lens)} 字，最长 {max(lens)} 字，平均 {sum(lens)//len(lens)} 字")


# ============ 5. 检索 + 生成 ============

_bm25 = None
_reranker = None


def get_bm25() -> BM25:
    global _bm25
    if _bm25 is None:
        store = get_store()
        _bm25 = BM25([m["text"] for m in store.meta])
    return _bm25


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker(mode=RERANK_MODE)
    return _reranker


def retrieve(question: str, top_k: int = TOP_K):
    """
    返回 [(文本, 元数据, 相关度), ...]，相关度越大越相关。

    两种模式：
      vector —— 只用向量检索（阶段 1 的行为）
      hybrid —— 向量 + BM25 两路粗筛，RRF 融合，再 CrossEncoder 精排（阶段 2）
    """
    store = get_store()
    if store.count() == 0:
        return []

    qvec = embed([question], is_query=True)[0]

    # ---------- 阶段 1 的行为，留着做对比 ----------
    if RETRIEVAL_MODE == "vector":
        return [
            (text, meta, 1.0 - dist)
            for text, meta, dist in store.query(qvec, top_k)
        ]

    # ---------- 粗筛：两路各取 CANDIDATE_K ----------
    n = min(CANDIDATE_K, store.count())
    vec_rank = [i for i, _t, _m, _d in store.query_with_index(qvec, n)]
    kw_rank = get_bm25().rank(question, n)

    # ---------- 融合 ----------
    fused = rrf_fuse([vec_rank, kw_rank])
    candidates = [(i, store.meta[i]["text"]) for i, _ in fused]

    # ---------- 精排 ----------
    if RERANK_MODE != "none":
        ranked = get_reranker().rerank(question, candidates, top_k)
        return [(text, store.meta[i], score) for i, text, score in ranked[:top_k]]

    return [(store.meta[i]["text"], store.meta[i], score) for i, score in fused[:top_k]]


def build_prompt(question: str, hits) -> str:
    context_parts = []
    for i, (doc, meta, _score) in enumerate(hits, 1):
        context_parts.append(
            f"【资料{i}｜来源：{meta['source']} 第{meta['chunk']}块】\n{doc}"
        )
    context = "\n\n".join(context_parts)

    return f"""你是一个严谨的文档问答助手。只能依据下面提供的资料回答问题。

规则：
1. 如果资料里没有答案，直接说"资料中未提及"，不要编造。
2. 回答时用【资料N】标注引用的来源。
3. 回答简洁，不要复述资料原文。

资料：
{context}

问题：{question}

回答："""


def answer(question: str, stream: bool = True) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    hits = retrieve(question)
    if not hits:
        print("库里还没有内容，先运行：python minimal_rag.py load 测试文档.txt")
        return ""

    prompt = build_prompt(question, hits)

    if not stream:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    # 流式输出
    print("\n回答：", end="", flush=True)
    collected = []
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=True,
    )
    for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            collected.append(delta)
    print()
    return "".join(collected)


# ============ 6. 命令行入口 ============

def main():
    args = sys.argv[1:]

    # 建库不需要 API Key，先让它能跑
    if args and args[0] == "load":
        for path in args[1:]:
            index_file(path)
        return

    if args and args[0] == "reset":
        get_store().clear()
        print("向量库已清空。")
        return

    if args and args[0] == "block":
        # 查看某一块的完整内容，用于检查分块质量（阶段 1 验收要用）
        store = get_store()
        if not store.meta:
            print("库是空的，先运行：python minimal_rag.py load 测试文档.txt")
            return
        if len(args) > 1:
            want = int(args[1])
            hit = [m for m in store.meta if m["chunk"] == want]
            if not hit:
                print(f"没有第 {want} 块，共 {len(store.meta)} 块")
                return
            m = hit[0]
            print(f"===== {m['source']} 第 {m['chunk']} 块  共 {len(m['text'])} 字 =====")
            print(m["text"])
            print("=" * 50)
        else:
            for m in store.meta:
                print(f"[第{m['chunk']}块] {len(m['text'])}字 | "
                      f"{m['text'][:100].replace(chr(10), ' ')}...")
        return

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("先设置环境变量：")
        print("  Windows:  set DEEPSEEK_API_KEY=sk-xxx")
        print("  bash:     export DEEPSEEK_API_KEY=sk-xxx")
        sys.exit(1)

    if args:
        question = " ".join(args)
        hits = retrieve(question)
        print(f"\n[分块={CHUNK_MODE}  检索={RETRIEVAL_MODE}  重排={RERANK_MODE}]")
        print(f"检索到 {len(hits)} 块：")
        for i, (doc, meta, score) in enumerate(hits, 1):
            preview = doc[:110].replace("\n", " ")
            print(f"  [{i}] 相关度={score:.4f}  第{meta['chunk']}块  {meta['source']}")
            print(f"      {preview}...")
        answer(question)
        return

    # 交互模式
    print("RAG 已就绪。输入问题，输入 q 退出。\n")
    while True:
        q = input("问题> ").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        answer(q)
        print()


if __name__ == "__main__":
    main()
