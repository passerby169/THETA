# =============================================================================
# THETA Backend Dockerfile
# =============================================================================
# 用途: 构建后端 FastAPI 服务镜像
# 使用方式:
#   docker build -t theta-backend:latest -f theta_project/Dockerfile .
#   docker-compose up -d
# =============================================================================

FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（OSS SDK 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有代码
COPY . .

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "asgi:app", "--host", "0.0.0.0", "--port", "8000"]
