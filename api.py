"""
HTTP 接口层 —— 把 RAG 从"一个脚本"变成"一个服务"。

为什么要这一层（面试会被问）：
  脚本只能在自己电脑上跑，服务才能被别人调用。
  公司里没有人会给你开终端，他们只会调接口。所以"能上线"和"能跑通"
  是两件不同的事，分界线就是这一层 HTTP 封装。

四个设计选择：

1) 为什么用 SSE 不用 WebSocket
   WebSocket 是双向协议，适合聊天室、协同编辑这种客户端要主动推消息的场景。
   这里只有服务端往客户端推 token，单向就够了。SSE 基于普通 HTTP，
   不用握手升级、不用额外端口、Nginx/网关直接支持、断线自动重连，
   实现成本比 WebSocket 低一个量级。

2) 为什么工具调用阶段不流式
   模型的 tool_calls 返回的是 JSON 字符串，必须等完整才能 json.loads。
   边收边解析会拿到半个 JSON，反而要写一堆容错。所以策略是：
   工具阶段用非流式跑完，最后一次生成再开 stream——用户感知到的延迟
   只有最后那一段，前面的工具耗时用进度事件补上体验。

3) 为什么 /search 单独开一个接口
   调试和演示都用得上。出问题时能立刻区分是"检索错了"还是"生成错了"，
   这是把所有逻辑塞进一个 /chat 里做不到的。

4) 为什么用 Pydantic 定义请求体
   自动做类型校验和参数范围限制，非法输入在进业务代码前就被挡掉，
   参数错误返回 422 并说明哪个字段错了，不用自己写 if 判断。

启动：
    export DEEPSEEK_API_KEY=sk-xxx      # 只用 /search 和 /health 可以不设
    uvicorn api:app --reload --port 8000

打开 http://127.0.0.1:8000/docs 就是自动生成的接文档，可以直接点着测。
"""

import json
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import metrics
import minimal_rag as rag
import tools as agent_tools

app = FastAPI(
    title="企业知识库问答 API",
    description="结构化分块 + BM25/向量混合检索 + LLM 重排 的 RAG 服务",
    version="1.0.0",
)


# ============ 请求 / 响应模型 ============

class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="查询问题")
    top_k: int = Field(4, ge=1, le=10, description="返回几块")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(4, ge=1, le=10)
    stream: bool = Field(True, description="是否流式返回")
    mode: str = Field("rag", description="rag = 固定检索链；agent = 模型自主调用工具")


class Chunk(BaseModel):
    chunk: int
    source: str
    score: float
    text: str


class SearchResponse(BaseModel):
    question: str
    elapsed_ms: float
    hits: list[Chunk]
    embedder: str


# ============ 工具函数 ============

def _need_key():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="未配置 DEEPSEEK_API_KEY。检索类接口（/search、/stats）不需要 Key，生成类接口需要。",
        )


def _sse(event: str, data) -> str:
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def _hits_to_chunks(hits):
    """
    统一转成 dict 而不是 Pydantic 对象。
    因为 SSE 分支要直接 json.dumps，而 json.dumps 不认识 BaseModel，
    会抛 TypeError；这个异常发生在第一个 yield 之前，表现是
    "HTTP 200 但 body 为空"，非常难查。所以入口就统一成 dict。
    """
    return [
        {
            "chunk": m["chunk"],
            "source": m["source"],
            "score": round(float(s), 4),
            "text": t,
        }
        for t, m, s in hits
    ]


# ============ 可观测性中间件 ============

def _client_ip(request: Request) -> str:
    """
    反向代理后面必读 X-Forwarded-For。

    Render / Nginx 这类部署里，request.client.host 永远是代理的 IP。
    不读这个头，所有访客会被算成同一个 IP —— 限流就从"防刷"变成
    "一个人访问 20 次，全站锁死"。这是上线后才暴露的典型问题。
    """
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def observe(request: Request, call_next):
    path = request.url.path

    # 健康检查和监控数据自己不计入统计、不消耗限流额度，
    # 否则 UptimeRobot 每分钟探活几次，就把正常用户的额度吃光了。
    if path in ("/health", "/admin/stats"):
        return await call_next(request)

    ip = _client_ip(request)
    if not metrics.allow(ip):
        metrics.record(path, 429, 0.0, ip, "限流")
        return JSONResponse(
            {"detail": f"请求过于频繁，每分钟最多 {metrics.RATE_LIMIT_PER_MIN} 次"},
            status_code=429,
        )

    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        metrics.record(path, 500, (time.perf_counter() - t0) * 1000, ip, "未捕获异常")
        raise

    elapsed = (time.perf_counter() - t0) * 1000
    metrics.record(path, response.status_code, elapsed, ip)
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.0f}"
    return response


# ============ 监控面板 ============

ADMIN_HTML = """<!doctype html>
<meta charset="utf-8"><title>RAG 服务监控</title>
<style>
 body{font:14px/1.6 -apple-system,"Segoe UI",sans-serif;margin:0;padding:24px;color:#1c1c1a;background:#f7f6f3}
 h1{font-size:17px;font-weight:600;margin:0 0 4px}
 .sub{color:#6b6a65;font-size:12px;margin-bottom:20px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
 .card{background:#fff;border:1px solid #e5e3dc;border-radius:10px;padding:14px 16px}
 .lab{font-size:12px;color:#6b6a65}
 .val{font-size:24px;font-weight:600;margin-top:2px}
 table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e5e3dc;border-radius:10px;overflow:hidden}
 th{text-align:left;font-size:12px;font-weight:500;color:#6b6a65;padding:8px 12px;border-bottom:1px solid #e5e3dc}
 td{padding:7px 12px;border-bottom:1px solid #f0eee8;font-size:13px}
 .ok{color:#3b6d11}.bad{color:#a32d2d}.warn{color:#854f0b}
 .mono{font-family:ui-monospace,Consolas,monospace}
</style>
<h1>企业知识库问答 · 服务监控</h1>
<div class="sub" id="sub">正在加载…</div>
<div class="grid">
  <div class="card"><div class="lab">总请求</div><div class="val" id="total">–</div></div>
  <div class="card"><div class="lab">P95 延迟</div><div class="val" id="p95">–</div></div>
  <div class="card"><div class="lab">错误率</div><div class="val" id="err">–</div></div>
  <div class="card"><div class="lab">累计成本</div><div class="val" id="cost">–</div></div>
</div>
<div class="grid">
  <div class="card"><div class="lab">输入 token</div><div class="val" id="tin">–</div></div>
  <div class="card"><div class="lab">输出 token</div><div class="val" id="tout">–</div></div>
  <div class="card"><div class="lab">模型调用</div><div class="val" id="calls">–</div></div>
  <div class="card"><div class="lab">向量化后端</div><div class="val" id="emb" style="font-size:15px;line-height:1.4">–</div></div>
</div>
<table><thead><tr><th>时间</th><th>接口</th><th>状态</th><th>耗时</th><th>来源 IP</th></tr></thead>
<tbody id="rows"><tr><td colspan="5" style="color:#6b6a65">暂无请求</td></tr></tbody></table>
<script>
const $=id=>document.getElementById(id);
async function tick(){
  try{
    const r=await fetch('/admin/stats');const d=await r.json();
    $('sub').textContent=`已运行 ${Math.round(d.uptime_sec/60)} 分钟 · 限流 ${d.rate_limit_per_min} 次/分钟/IP · 知识库 ${d.chunks} 块`;
    $('total').textContent=d.requests.total;
    $('p95').textContent=d.latency_ms.p95+' ms';
    const e=d.requests.total?(d.requests.error/d.requests.total*100):0;
    $('err').textContent=e.toFixed(1)+' %';
    $('err').className='val '+(e>5?'bad':'ok');
    $('cost').textContent=d.price_configured?('¥'+d.cost_cny.toFixed(3)):'未配单价';
    $('cost').style.fontSize=d.price_configured?'24px':'14px';
    $('tin').textContent=d.tokens.prompt;
    $('tout').textContent=d.tokens.completion;
    $('calls').textContent=d.tokens.llm_calls;
    $('emb').textContent=d.embedder;
    $('rows').innerHTML=d.recent.length?d.recent.slice().reverse().map(x=>
      `<tr><td class="mono">${new Date(x.t*1000).toLocaleTimeString()}</td>
       <td class="mono">${x.path}</td>
       <td class="${x.status<400?'ok':x.status==429?'warn':'bad'}">${x.status}</td>
       <td class="mono">${x.ms} ms</td>
       <td class="mono">${x.ip}</td></tr>`).join(''):'<tr><td colspan="5">暂无请求</td></tr>';
  }catch(e){$('sub').textContent='监控数据读取失败：'+e;}
}
tick();setInterval(tick,3000);
</script>
"""


@app.get("/admin", response_class=HTMLResponse, summary="监控面板")
def admin_page():
    """给运维和演示用的可视化面板，和 /docs 互补：那个测功能，这个看运行状态。"""
    return ADMIN_HTML


@app.get("/admin/stats", summary="监控数据（JSON）")
def admin_stats():
    store = rag.get_store()
    embedder = rag.get_embedder()
    return metrics.snapshot({
        "chunks": store.count(),
        "embedder": rag.embedder_name(),
        "embed_api_calls": getattr(embedder, "calls", 0),
        "degraded": rag._degraded,
    })


# ============ 接口 ============

@app.get("/", summary="接口索引")
def root():
    """根路径给个索引，避免访问 127.0.0.1:8000 看到 404 以为服务没起来。"""
    return {
        "service": "企业知识库问答 API",
        "docs": "/docs",
        "endpoints": {
            "GET  /health": "健康检查",
            "GET  /stats": "知识库统计",
            "POST /search": "只检索不生成",
            "POST /chat": "固定 RAG（非流式）",
            "POST /chat/stream": "固定 RAG（SSE 流式）",
            "POST /agent/stream": "Agentic RAG（SSE 流式）",
            "POST /documents": "按服务器本地路径入库",
        },
    }


@app.get("/health", summary="健康检查")
def health():
    """部署后第一件事就是打这个接口，确认服务活着、库里有内容。"""
    store = rag.get_store()
    return {
        "status": "ok",
        "chunks": store.count(),
        "api_key": bool(os.environ.get("DEEPSEEK_API_KEY")),
    }


@app.get("/stats", summary="知识库统计")
def stats():
    store = rag.get_store()
    sources = {}
    for m in store.meta:
        sources[m["source"]] = sources.get(m["source"], 0) + 1
    return {
        "chunks": store.count(),
        "documents": sources,
        "embedder": rag.embedder_name(),
        "config": {
            "chunk_mode": rag.CHUNK_MODE,
            "retrieval_mode": rag.RETRIEVAL_MODE,
            "rerank_mode": rag.RERANK_MODE,
            "candidate_k": rag.CANDIDATE_K,
            "top_k": rag.TOP_K,
        },
        "degraded": rag._degraded,
    }


@app.post("/search", response_model=SearchResponse, summary="只检索不生成")
def search(req: SearchRequest):
    """
    单独暴露检索，是给调试用的。
    回答错了，先打这个接口看检索结果——90% 的问题出在这一层。
    """
    if rag.get_store().count() == 0:
        raise HTTPException(400, "知识库为空，先 POST /documents 或运行 load 命令建库。")

    t0 = time.perf_counter()
    hits = rag.retrieve(req.question, req.top_k)
    elapsed = (time.perf_counter() - t0) * 1000

    return SearchResponse(
        question=req.question,
        elapsed_ms=round(elapsed, 1),
        hits=_hits_to_chunks(hits),
        embedder=rag.embedder_name(),
    )


@app.post("/chat", summary="固定 RAG 问答（非流式）")
def chat(req: ChatRequest):
    _need_key()
    if rag.get_store().count() == 0:
        raise HTTPException(400, "知识库为空。")

    t0 = time.perf_counter()
    hits = rag.retrieve(req.question, req.top_k)
    t_retrieve = (time.perf_counter() - t0) * 1000

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com")
    resp = client.chat.completions.create(
        model=rag.LLM_MODEL,
        messages=[{"role": "user", "content": rag.build_prompt(req.question, hits)}],
        temperature=0.2,
    )
    metrics.record_tokens(resp.usage.prompt_tokens, resp.usage.completion_tokens)

    return {
        "question": req.question,
        "answer": resp.choices[0].message.content,
        "citations": _hits_to_chunks(hits),
        "timing": {
            "retrieve_ms": round(t_retrieve, 1),
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        },
    }


@app.post("/chat/stream", summary="固定 RAG 问答（SSE 流式）")
def chat_stream(req: ChatRequest):
    _need_key()
    if rag.get_store().count() == 0:
        raise HTTPException(400, "知识库为空。")

    def gen():
        t0 = time.perf_counter()
        try:
            hits = rag.retrieve(req.question, req.top_k)
            yield _sse("citations", _hits_to_chunks(hits))
            yield _sse("timing", {"retrieve_ms": round((time.perf_counter() - t0) * 1000, 1)})

            from openai import OpenAI
            client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                            base_url="https://api.deepseek.com")
            stream = client.chat.completions.create(
                model=rag.LLM_MODEL,
                messages=[{"role": "user", "content": rag.build_prompt(req.question, hits)}],
                temperature=0.2,
                stream=True,
                # 流式默认拿不到 token 用量，必须显式要。
                # 开了之后最后一个 chunk 只带 usage、choices 是空列表，
                # 直接 chunk.choices[0] 会 IndexError —— 这是加计费时最容易踩的坑。
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if not chunk.choices:
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        metrics.record_tokens(usage.prompt_tokens, usage.completion_tokens)
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield _sse("token", delta)
        except Exception as e:
            # 流已经开始（状态码已发 200）就无法再改成 500，只能把错误当事件发出去，
            # 让前端能显示"出错了"而不是一直转圈。这是 SSE 的固有限制。
            yield _sse("error", f"{type(e).__name__}: {e}")
        finally:
            yield _sse("done", {"total_ms": round((time.perf_counter() - t0) * 1000, 1)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/agent/stream", summary="Agentic RAG：模型自主调用工具（SSE）")
def agent_stream(req: ChatRequest):
    """
    事件流：tool_call → tool_result → token... → done
    前端可以据此显示"正在检索资料…"这类进度，比转圈有信息量。
    """
    _need_key()
    if rag.get_store().count() == 0:
        raise HTTPException(400, "知识库为空。")

    def gen():
        t0 = time.perf_counter()
        try:
            for ev in agent_tools.run_agent(req.question):
                if ev["type"] == "done":
                    ev["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                yield _sse(ev["type"], ev)
        except Exception as e:
            yield _sse("error", f"{type(e).__name__}: {e}")

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/documents", summary="上传文档并入库")
def upload(path: str):
    """
    这里故意只接受服务器本地路径，不接受文件上传。
    真做上传要处理：临时文件、病毒扫描、格式白名单、大小限制、
    超时清理——都是能写一篇文章的工程量，对现在的目标没有帮助。
    """
    if not os.path.exists(path):
        raise HTTPException(404, f"文件不存在：{path}")
    rag.index_file(path)
    return {"indexed": path, "total_chunks": rag.get_store().count()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
