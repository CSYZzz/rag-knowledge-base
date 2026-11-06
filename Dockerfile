# 企业知识库问答服务 —— 线上镜像
#
# 三个刻意的选择，每个都是为了塞进 512MB 内存：
#   1. slim 基础镜像，不用 full（full 多带 500MB 编译工具链，运行时一点用不上）
#   2. requirements.prod.txt 里没有 torch（这是最关键的一条）
#   3. --workers 1 单进程（metrics 和索引都在进程内存里，多 worker 各算各的，
#      监控数据会变成"N 份统计"而不是一份，索引也会重复加载 N 份）
#
# 镜像体积：约 250MB（对比带 torch 的版本约 2GB）

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBED_BACKEND=remote \
    PORT=10000

WORKDIR /app

# 先拷依赖清单、再拷代码：改业务代码不会让依赖层缓存失效，
# 重建时间从约 3 分钟降到约 20 秒（免费层构建分钟数有限，这条不是洁癖）
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

COPY . .

EXPOSE 10000

HEALTHCHECK --interval=60s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','10000')+'/health',timeout=4)"

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
