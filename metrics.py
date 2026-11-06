"""
可观测性与保护 —— 让这个服务「能被看见」。

为什么部署这一个动作，价值比再写一个项目高：
  跑在你自己电脑上的服务，出问题你能看终端。跑在云上没人管的服务，
  出问题你什么都不知道。所以在把服务推上线之前，先得给它装上仪表盘。

这一层只做四件事，都是运维面试会盯着问的：

  1. 请求日志    谁、什么时候、问了什么、耗时多少、成没成功
  2. 分位数延迟  平均延迟没有意义（被少数快请求拉低），要看 P95
  3. 成本核算    每次问答烧了多少 token、折合多少钱
  4. 限流        公网服务不设限流，等于把 API Key 交出去一半

为什么不用 Prometheus / OpenTelemetry：
  免费层 512MB 内存，起一个 sidecar 就没了；
  而且这个规模下，一个内存里的 deque 完全够用。
  「在资源约束下做正确的取舍」本身就是这一层要展示的东西。
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

# 最近多少条请求参与分位数计算。太多了吃内存，太少了 P95 不稳。
RECENT_MAX = 300

# 限流窗口与默认阈值（每分钟每 IP）
WINDOW_SEC = 60
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "20"))

# 单价（元 / 百万 token），从环境变量读，默认 0 表示未配置。
# 不写死是因为模型价格会变，写死的数字过两个月就是错的，而错的数字比没有更糟。
PRICE_IN = float(os.environ.get("PRICE_IN_PER_M", "0"))
PRICE_OUT = float(os.environ.get("PRICE_OUT_PER_M", "0"))

_lock = threading.Lock()

_recent: deque[dict] = deque(maxlen=RECENT_MAX)
_counters = {"total": 0, "ok": 0, "error": 0, "rate_limited": 0}
_tokens = {"prompt": 0, "completion": 0, "llm_calls": 0}

# ip -> 该 ip 最近的请求时间戳
_ip_hits: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=200))

_started_at = time.time()


# ============ 记录 ============

def record(path: str, status: int, ms: float, ip: str, note: str = "") -> None:
    with _lock:
        _counters["total"] += 1
        if status >= 500:
            _counters["error"] += 1
        elif status == 429:
            _counters["rate_limited"] += 1
        else:
            _counters["ok"] += 1

        _recent.append({
            "t": round(time.time(), 1),
            "path": path,
            "status": status,
            "ms": round(ms, 1),
            "ip": ip,
            "note": note,
        })


def record_tokens(prompt: int, completion: int) -> None:
    """在模型调用处上报真实 token 数。流式要开 include_usage 才拿得到。"""
    if not prompt and not completion:
        return
    with _lock:
        _tokens["prompt"] += int(prompt or 0)
        _tokens["completion"] += int(completion or 0)
        _tokens["llm_calls"] += 1


# ============ 限流 ============

def allow(ip: str) -> bool:
    """滑动窗口：只统计最近 60 秒内的请求数。"""
    now = time.time()
    with _lock:
        hits = _ip_hits[ip]
        while hits and now - hits[0] > WINDOW_SEC:
            hits.popleft()
        if len(hits) >= RATE_LIMIT_PER_MIN:
            return False
        hits.append(now)
        return True


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * p / 100), len(ordered) - 1)
    return ordered[idx]


# ============ 快照 ============

def snapshot(extra: dict | None = None) -> dict:
    with _lock:
        latencies = [r["ms"] for r in _recent if r["status"] < 500]
        cost = (
            _tokens["prompt"] / 1_000_000 * PRICE_IN
            + _tokens["completion"] / 1_000_000 * PRICE_OUT
        )
        data = {
            "uptime_sec": round(time.time() - _started_at, 1),
            "requests": dict(_counters),
            "latency_ms": {
                "p50": round(_percentile(latencies, 50), 1),
                "p95": round(_percentile(latencies, 95), 1),
                "max": round(max(latencies), 1) if latencies else 0.0,
                "samples": len(latencies),
            },
            "tokens": dict(_tokens),
            "cost_cny": round(cost, 4),
            "price_configured": bool(PRICE_IN or PRICE_OUT),
            "rate_limit_per_min": RATE_LIMIT_PER_MIN,
            "recent": list(_recent)[-20:],
        }
    if extra:
        data.update(extra)
    return data


def reset() -> None:
    """评测和压测前清空，避免把测试流量算进线上统计。"""
    with _lock:
        _recent.clear()
        _counters.update({"total": 0, "ok": 0, "error": 0, "rate_limited": 0})
        _tokens.update({"prompt": 0, "completion": 0, "llm_calls": 0})
        _ip_hits.clear()
