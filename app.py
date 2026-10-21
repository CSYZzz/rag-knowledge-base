"""
Streamlit 界面 —— 让不懂技术的人也能用。

为什么最后才做界面（而不是先做）：
  界面不改变答案质量，只改变"别人愿不愿意试一下"。
  检索链路没调好之前做界面，等于把一个不准的东西包得更好看。
  但做界面有它自己的价值：它能录成 60 秒视频，而视频是简历上
  唯一能让人在 30 秒内理解你做了什么的东西。

启动：
    streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st

import minimal_rag as rag
import tools as agent_tools

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="企业知识库问答", page_icon="📚", layout="wide")


# ============ 侧边栏：配置与建库 ============

with st.sidebar:
    st.title("📚 知识库问答")

    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        value=os.environ.get("DEEPSEEK_API_KEY", ""),
        help="只填在页面上，不写进代码也不上传。只用检索预览不需要填。",
    )
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key

    st.divider()

    mode = st.radio(
        "检索模式",
        ["固定 RAG", "Agent（模型自主检索）"],
        help="固定 RAG 快且稳；Agent 会自己决定查几次、换什么词，复杂问题更准但更慢。",
    )

    top_k = st.slider("引用块数", 1, 8, 4)
    show_debug = st.checkbox("显示检索明细", value=True)

    st.divider()

    st.subheader("文档入库")
    uploaded = st.file_uploader("上传 txt / md / pdf", type=["txt", "md", "pdf"])
    if uploaded is not None and st.button("加入知识库", use_container_width=True):
        dest = UPLOAD_DIR / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        with st.spinner("正在解析、分块、向量化…"):
            rag.index_file(str(dest))
        st.success(f"已入库：{uploaded.name}")
        st.rerun()

    if st.button("清空知识库", use_container_width=True):
        rag.get_store().clear()
        st.warning("已清空")
        st.rerun()

    st.divider()

    store = rag.get_store()
    c1, c2 = st.columns(2)
    c1.metric("文档块", store.count())
    c2.metric("向量维度", store.vectors.shape[1] if store.vectors is not None else 0)
    st.caption(f"Embedding：{rag.embedder_name()}")
    if rag._degraded:
        st.warning("BGE 模型未加载，正在用哈希降级检索，效果会差很多。")


# ============ 主区：对话 ============

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("企业知识库问答")
st.caption("基于结构化分块 + BM25/向量混合检索 + LLM 重排。回答会标注引用了哪一块。")

if rag.get_store().count() == 0:
    st.info("知识库还是空的。在左侧上传一份文档，或者先在终端跑 `python minimal_rag.py load 测试文档.txt`。")


def render_citations(hits):
    if not show_debug or not hits:
        return
    with st.expander(f"📎 引用来源（{len(hits)} 块）"):
        for i, (text, meta, score) in enumerate(hits, 1):
            st.markdown(
                f"**{i}. 第 {meta['chunk']} 块** · {meta['source']} · "
                f"相关度 `{score:.4f}` · {len(text)} 字"
            )
            st.text(text[:600] + ("…" if len(text) > 600 else ""))
            st.divider()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            render_citations(msg["hits"])


question = st.chat_input("问点什么，比如：保修期是多久？")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        has_key = bool(os.environ.get("DEEPSEEK_API_KEY"))
        hits = rag.retrieve(question, top_k)
        placeholder = st.empty()

        # ---------- 没有 API Key：只展示检索结果 ----------
        if not has_key:
            placeholder.info("未填 API Key，下面只显示检索结果，不生成回答。")
            render_citations(hits)
            st.session_state.messages.append({
                "role": "assistant",
                "content": "（未配置 API Key，仅展示检索结果）",
                "hits": hits,
            })
            st.stop()

        # ---------- 固定 RAG ----------
        if mode == "固定 RAG":
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                            base_url="https://api.deepseek.com")
            stream = client.chat.completions.create(
                model=rag.LLM_MODEL,
                messages=[{"role": "user",
                           "content": rag.build_prompt(question, hits)}],
                temperature=0.2,
                stream=True,
            )
            buf = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    buf.append(delta)
                    placeholder.markdown("".join(buf) + "▌")
            answer = "".join(buf)
            placeholder.markdown(answer)
            render_citations(hits)

            st.session_state.messages.append({
                "role": "assistant", "content": answer, "hits": hits,
            })

        # ---------- Agent 模式 ----------
        else:
            trace = []
            answer_parts = []
            with st.status("模型正在自行检索…", expanded=True) as status:
                for ev in agent_tools.run_agent(question):
                    if ev["type"] == "tool_call":
                        st.write(f"→ 调用 `{ev['name']}`：`{ev['args']}`")
                    elif ev["type"] == "tool_result":
                        st.write(f"← {ev['preview']}…")
                    elif ev["type"] == "token":
                        answer_parts.append(ev["text"])
                        placeholder.markdown("".join(answer_parts) + "▌")
                    elif ev["type"] == "done":
                        trace = ev.get("tools_used", [])
                        status.update(
                            label=f"完成（{ev['turns']} 轮，调用工具 {len(trace)} 次）",
                            state="complete", expanded=False,
                        )

            answer = "".join(answer_parts)
            placeholder.markdown(answer)
            render_citations(hits)
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "hits": hits,
                "trace": trace,
            })
