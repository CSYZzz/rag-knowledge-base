"""
Function Calling 工具层 —— 让模型自己决定"检索什么、检索几次"。

为什么加这一层：

  固定 RAG 的流程是"问题进来 → 检索 4 块 → 拼 Prompt → 生成"，
  检索这一步是写死的：不管问题需不需要查资料，都要查一次；
  也不管第一次查得准不准，都没有第二次机会。

  Agentic RAG 把"要不要查、查什么、查几轮"交给模型自己判断：

      问题 → 模型思考 → 调用 search_knowledge("保修期") → 看结果
                     → 觉得不够，再调 search_knowledge("质保年限") → 看结果
                     → 够了，回答

  代价是更慢更贵，收益是复杂问题（多跳、需要换词重试）答得更准。
  所以默认还是固定 RAG，这里是可切换的一条增强路径。

三个工具的取舍和真实 Agent 系统一致：
  search_knowledge  先搜（不知道去哪儿找的时候用）
  list_sections     再看目录（知道大概在哪一章，但不知道具体块）
  fetch_chunk       最后读原文（知道是哪一块了，要全文）

用法（命令行）：
    python tools.py "保修期是多久"
"""

import json
import os
import re
import sys

import minimal_rag as rag


# ============ 工具定义（给模型看的说明书） ============

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "在企业知识库中检索与问题相关的文档片段。"
                "返回若干块，每块含块号、所属文档、相关度和正文。"
                "如果第一次检索的结果不相关，可以换一个关键词再调用一次。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索用的关键词或短句，不要直接传整段问题",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回几块，默认 4，建议 3~8",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sections",
            "description": (
                "列出知识库中所有文档的章节标题和块号。"
                "当你不确定答案在哪一章时，先看目录再定位，比盲目检索更省时间。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_chunk",
            "description": "按块号读取某一块的完整正文。用于检索结果被截断、或需要看全文才能判断时。",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "integer", "description": "块号"},
                },
                "required": ["chunk_id"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是一个严谨的企业知识库问答助手，可以调用工具查阅内部资料。

工作方式：
1. 先判断是否需要查资料。凡是涉及文档内容的问题，必须先调用工具，不要凭记忆回答。
2. 不确定答案在哪一章时，先 list_sections 看目录，再用 search_knowledge 定位。
3. 第一次检索没找到时，换一种说法或关键词再试，最多再试两次。
4. 找到后基于原文回答，并用【第N块】标注依据。
5. 资料里确实没有的，直接回答"资料中未提及"，不要编造。

回答要简洁，不要复述整段原文。"""


# ============ 工具实现 ============

def search_knowledge(query: str, top_k: int = 4) -> str:
    hits = rag.retrieve(query, top_k=max(1, min(int(top_k), 10)))
    if not hits:
        return "知识库为空或没有检索到内容。"

    lines = [f"检索「{query}」，命中 {len(hits)} 块："]
    for doc, meta, score in hits:
        lines.append(
            f"\n【第{meta['chunk']}块｜{meta['source']}｜相关度 {score:.4f}】\n{doc}"
        )
    return "\n".join(lines)


def list_sections() -> str:
    store = rag.get_store()
    if store.count() == 0:
        return "知识库为空，请先建库。"

    # 分块时给每块前置了「【章节名】」，这里把它抽出来当目录
    by_source = {}
    for m in store.meta:
        title = ""
        first = m["text"].split("\n", 1)[0]
        mt = re.match(r"^【(.+?)】$", first.strip())
        if mt:
            title = mt.group(1)
        by_source.setdefault(m["source"], []).append((m["chunk"], title or "（无标题）"))

    lines = []
    for source, items in by_source.items():
        lines.append(f"{source}（共 {len(items)} 块）")
        for cid, title in items:
            lines.append(f"  第{cid}块  {title}")
    return "\n".join(lines)


def fetch_chunk(chunk_id: int) -> str:
    store = rag.get_store()
    hit = [m for m in store.meta if m["chunk"] == int(chunk_id)]
    if not hit:
        return f"没有第 {chunk_id} 块，当前共 {store.count()} 块。"
    m = hit[0]
    return f"【第{m['chunk']}块｜{m['source']}｜{len(m['text'])} 字】\n{m['text']}"


_DISPATCH = {
    "search_knowledge": lambda a: search_knowledge(a.get("query", ""), a.get("top_k", 4)),
    "list_sections": lambda a: list_sections(),
    "fetch_chunk": lambda a: fetch_chunk(a.get("chunk_id", 0)),
}


def execute_tool(name: str, arguments: str) -> str:
    """模型给的是 JSON 字符串，解析失败也要能降级，不能让整条链断掉。"""
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"未知工具：{name}"
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return f"参数不是合法 JSON：{arguments[:80]}"
    try:
        return fn(args)
    except Exception as e:
        return f"工具执行出错：{type(e).__name__}: {e}"


# ============ Agent 循环 ============

def _client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )


def run_agent(question: str, max_turns: int = 4, model: str = "deepseek-chat"):
    """
    产出事件的生成器。事件类型：
        {"type": "tool_call",   "name": ..., "args": ...}
        {"type": "tool_result", "name": ..., "preview": ...}
        {"type": "token",       "text": ...}
        {"type": "done",        "answer": ..., "turns": n, "tools_used": [...]}

    为什么要用生成器：接口层要一边跑一边把进度推给前端（SSE），
    如果用普通函数，前端就只能干等。生成器让"边算边播"变成一行 for 循环。
    """
    client = _client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools_used = []

    for turn in range(1, max_turns + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMA,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        # 没有工具调用 = 模型认为可以答了，进入最后的生成阶段
        if not tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            tools_used.append(name)
            yield {"type": "tool_call", "name": name, "args": args}

            result = execute_tool(name, args)
            yield {"type": "tool_result", "name": name,
                   "preview": result[:160].replace("\n", " ")}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # 工具阶段结束。这时再开一次 stream=True，只有这一步是流式的，
    # 因为工具调用必须等完整 JSON 才能往下走，流式反而更慢。
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        stream=True,
        # 同上：要计费就得显式要 usage，且末尾 chunk 的 choices 为空要先判空
        stream_options={"include_usage": True},
    )
    collected = []
    for chunk in stream:
        if not chunk.choices:
            usage = getattr(chunk, "usage", None)
            if usage:
                try:
                    import metrics
                    metrics.record_tokens(usage.prompt_tokens, usage.completion_tokens)
                except Exception:
                    pass
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            collected.append(delta)
            yield {"type": "token", "text": delta}

    yield {
        "type": "done",
        "answer": "".join(collected),
        "turns": turn,
        "tools_used": tools_used,
    }


def answer(question: str) -> str:
    """命令行用的简化版：只打印过程和最终答案。"""
    print(f"\n问题：{question}\n")
    final = ""
    for ev in run_agent(question):
        if ev["type"] == "tool_call":
            print(f"  → 调用 {ev['name']}({ev['args']})")
        elif ev["type"] == "tool_result":
            print(f"    ← {ev['preview']}...")
        elif ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "done":
            final = ev["answer"]
            print(f"\n\n[共 {ev['turns']} 轮，调用工具 {len(ev['tools_used'])} 次]")
    return final


if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("先设置环境变量：export DEEPSEEK_API_KEY=sk-xxx")
        sys.exit(1)
    q = " ".join(sys.argv[1:]) or "保修期是多久"
    answer(q)
