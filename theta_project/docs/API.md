# THETA Platform API Documentation

> 本文档基于 `main.py` 中的实际 API 端点自动生成，完整配置文件见 `config/api_endpoints.yaml`

---

## 基础信息

| 项目 | 值 |
|------|-----|
| **Base URL (Dev)** | `http://localhost:8000` |
| **Base URL (Prod)** | `https://theta.code-soul.com` |
| **API 文档** | `/docs` (Swagger) / `/redoc` (ReDoc) |
| **认证方式** | Bearer Token (JWT) |

### 认证流程

1. 注册：`POST /api/auth/register`
2. 登录：`POST /api/auth/login` → 获取 `access_token`
3. 请求时在 Header 中附加：`Authorization: Bearer <access_token>`

---

## 1. 认证模块 `/api/auth`

### `POST /api/auth/register` — 用户注册

**认证**: 不需要

**请求体**:
```json
{
  "username": "zhangsan",
  "email": "zhangsan@example.com",
  "password": "secure_password"
}
```

**响应** `201 Created`:
```json
{
  "id": 1,
  "username": "zhangsan",
  "email": "zhangsan@example.com",
  "is_active": true
}
```

**错误**:
- `400`: 用户名或邮箱已注册

---

### `POST /api/auth/login` — 用户登录

**认证**: 不需要

**请求**: `application/x-www-form-urlencoded`

```
username=zhangsan&password=secure_password
```

**响应** `200 OK`:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**错误**:
- `401`: 用户名或密码错误

---

## 2. 文件上传模块 `/api/upload`

### `POST /api/upload` — 上传文件（后端本地存储）

**认证**: 需要

**请求**: `multipart/form-data`

| 参数 | 类型 | 说明 |
|------|------|------|
| `file` | File | 要上传的文件 |

**响应** `201 Created`:
```json
{
  "id": 1,
  "filename": "data.csv",
  "file_path": "/path/to/local/file",
  "file_type": "uploads",
  "created_at": "2026-04-25T10:00:00Z"
}
```

---

### `POST /api/upload/complete` — 通知上传完成

**认证**: 需要

**请求体**:
```json
{
  "dataset_name": "my_research",
  "filename": "data.csv",
  "oss_path": "raw_data/zhangsan/my_research/data.csv",
  "file_size": 1024000
}
```

**响应** `201 Created`: 见 `POST /api/upload`

---

### `GET /api/files` — 获取文件列表

**认证**: 需要

**响应** `200 OK`:
```json
[
  {
    "id": 1,
    "filename": "data.csv",
    "file_path": "raw_data/zhangsan/my_research/data.csv",
    "file_type": "oss_upload",
    "created_at": "2026-04-25T10:00:00Z",
    "dataset_name": "my_research"
  }
]
```

---

### `GET /api/oss/sts-token` — 获取 STS 临时凭证

**认证**: 需要

**查询参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `dataset_name` | string | 数据集名称 |

**响应** `200 OK`:
```json
{
  "credentials": {
    "access_key_id": "STS.xxx",
    "access_key_secret": "xxx",
    "security_token": "xxx",
    "expiration": "2026-04-25T12:00:00Z"
  },
  "upload_path": "raw_data/zhangsan/my_research/",
  "bucket": "theta-prod-20260123",
  "endpoint": "oss-cn-shanghai.aliyuncs.com",
  "region": "cn-shanghai"
}
```

> 前端使用此凭证直接上传文件到 OSS，无需经过后端。

---

## 3. 预处理模块 `/api/preprocessing`

### `GET /api/preprocessing/check/{dataset}` — 检查预处理状态

**认证**: 需要

**路径参数**:
- `dataset`: 数据集名称

**响应** `200 OK`:
```json
{
  "dataset": "my_research",
  "has_bow": true,
  "has_embeddings": true,
  "ready_for_training": true,
  "bow_path": "raw_data/zhangsan/my_research/workspace/bow_matrix.npy",
  "embedding_path": "raw_data/zhangsan/my_research/workspace/embeddings.npy",
  "vocab_path": "raw_data/zhangsan/my_research/workspace/vocab.json"
}
```

---

### `POST /api/preprocessing/start` — 启动预处理

**认证**: 需要

**请求体**:
```json
{
  "dataset": "my_research",
  "text_column": "content"
}
```

**响应** `201 Created`:
```json
{
  "job_id": "prep_abc123def456",
  "dataset": "my_research",
  "status": "completed",
  "progress": 100,
  "message": "预处理已完成（BOW 和嵌入将在训练时自动生成）"
}
```

---

### `GET /api/preprocessing/{job_id}` — 获取预处理任务状态

**认证**: 需要

**响应** `200 OK`:
```json
{
  "job_id": "prep_abc123def456",
  "dataset": "my_research",
  "status": "completed",
  "progress": 100,
  "message": "任务已完成或不存在"
}
```

---

## 4. 训练任务模块 `/api/train`

### `POST /api/train/start` — 提交训练任务

**认证**: 需要

**请求体**:
```json
{
  "file_id": 1,
  "dataset_name": "my_research",
  "model_type": "theta",
  "model_size": "0.6B",
  "mode": "zero_shot",
  "num_topics": 20,
  "epochs": 100,
  "batch_size": 64,
  "learning_rate": 0.002,
  "hidden_dim": 512,
  "patience": 10,
  "vocab_size": 5000,
  "language": "chinese"
}
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_type` | `theta` | 模型类型：`theta` / `lda` / `bertopic` / `ctm` 等 |
| `model_size` | `0.6B` | Qwen 模型规格：`0.6B` / `4B` / `8B`（仅 theta 使用） |
| `mode` | `zero_shot` | 嵌入模式：`zero_shot` / `supervised` / `unsupervised` |
| `num_topics` | `20` | 主题数量 |
| `epochs` | `100` | 训练轮数 |
| `language` | `chinese` | 数据语言：`chinese` / `english` |

**响应** `201 Created`:
```json
{
  "id": 1,
  "user_id": 1,
  "status": "running",
  "dlc_job_id": "DLC-12345678",
  "created_at": "2026-04-25T10:00:00Z"
}
```

---

### `GET /api/train/{job_id}/status` — 查询训练状态

**认证**: 需要

**响应** `200 OK`:
```json
{
  "job_id": 1,
  "status": "running",
  "dlc_job_id": "DLC-12345678",
  "message": "训练进行中",
  "created_at": "2026-04-25T10:00:00Z",
  "error_message": null
}
```

**任务状态**:
| 状态 | 说明 |
|------|------|
| `pending` | 等待调度 |
| `running` | 训练进行中 |
| `succeeded` | 训练完成 |
| `failed` | 训练失败 |

---

### `GET /api/train/{job_id}/metrics` — 获取训练指标

**认证**: 需要

**响应** `200 OK`:
```json
{
  "job_id": 1,
  "status": "running",
  "epochs": [1, 2, 3, 4, 5],
  "loss": [2.5, 2.1, 1.8, 1.5, 1.2],
  "accuracy": [0.3, 0.45, 0.58, 0.68, 0.75]
}
```

---

### `GET /api/train/{job_id}/summary` — 获取训练摘要

**认证**: 需要

**响应** `200 OK`:
```json
{
  "job_id": 1,
  "status": "succeeded",
  "summary": "训练完成，最终 Loss: 0.85，Accuracy: 0.78"
}
```

---

### `GET /api/train/jobs` — 获取训练任务列表

**认证**: 需要

**响应** `200 OK`:
```json
[
  {
    "id": 2,
    "user_id": 1,
    "status": "succeeded",
    "dlc_job_id": "DLC-87654321",
    "created_at": "2026-04-25T10:00:00Z"
  }
]
```

---

### `POST /api/train/callback` — DLC 回调通知

**认证**: 不需要（使用 `secret_key` 校验）

**请求体**:
```json
{
  "job_id": 1,
  "status": "succeeded",
  "secret_key": "theta-at-2026-production-key"
}
```

**响应** `200 OK`:
```json
{
  "success": true,
  "message": "Job 1 status updated to succeeded"
}
```

---

## 5. 数据查询模块 `/api/data` 和 `/api/results`

### `GET /api/data/oss-datasets` — 获取有结果的数据集列表

**认证**: 需要

**响应** `200 OK`:
```json
{
  "datasets": [
    {"name": "my_research", "chart_count": 8}
  ]
}
```

---

### `GET /api/results/{dataset}/models` — 获取可用模型列表

**认证**: 需要

**响应** `200 OK`:
```json
{
  "dataset": "my_research",
  "models": ["theta", "nvdm", "bertopic", "lda"]
}
```

---

### `GET /api/results/{dataset}/topic-words` — 获取主题词

**认证**: 需要

**查询参数**:
- `model`: 模型类型（默认 `theta`）

**响应** `200 OK`:
```json
{
  "dataset": "my_research",
  "model": "theta",
  "topics": {
    "0": [["机器学习", 0.85], ["深度学习", 0.72]],
    "1": [["数据分析", 0.90], ["可视化", 0.68]]
  }
}
```

---

### `GET /api/results/{dataset}/metrics` — 获取评估指标

**认证**: 需要

**响应** `200 OK`:
```json
{
  "dataset": "my_research",
  "model": "theta",
  "metrics": {
    "TD": 0.85,
    "iRBO": 0.72,
    "NPMI": 0.65,
    "C_V": 0.58,
    "UMass": 0.42,
    "Exclusivity": 0.91,
    "PPL": 38.5
  }
}
```

**指标说明**:
| 指标 | 全称 | 方向 |
|------|------|------|
| `TD` | Topic Diversity | 越高越好 |
| `iRBO` | Inverse Rank-Biased Overlap | 越高越好 |
| `NPMI` | Normalized PMI | 越高越好 |
| `C_V` | C_V Coherence | 越高越好 |
| `UMass` | UMass Coherence | 越高越好 |
| `Exclusivity` | Topic Exclusivity | 越高越好 |
| `PPL` | Perplexity | 越低越好 |

---

### `GET /api/results/{dataset}/visualizations` — 获取可视化文件列表

**认证**: 需要

**响应** `200 OK`:
```json
{
  "dataset": "my_research",
  "model": "theta",
  "global_files": [
    {"name": "topic_distribution.png", "path": "global/topic_distribution.png", "url": "https://...", "size": 102400, "type": "global"}
  ],
  "topic_files": {
    "0": [{"name": "wordcloud.png", "path": "topic/topic_0/wordcloud.png", "url": "https://...", "size": 51200, "type": "topic"}],
    "1": [{"name": "wordcloud.png", "path": "topic/topic_1/wordcloud.png", "url": "https://...", "size": 51200, "type": "topic"}]
  }
}
```

---

### `DELETE /api/datasets/{dataset}` — 删除数据集

**认证**: 需要

**响应** `200 OK`:
```json
{
  "success": true,
  "message": "数据集 my_research 已删除"
}
```

---

## 6. AI 对话模块 `/api/agent` 和 `/api/chat`

### `POST /api/agent/chat` — AI 对话

**认证**: 不需要（开发模式自动使用测试用户）

**请求体**:
```json
{
  "message": "分析一下这个数据集的主题分布",
  "session_id": "default",
  "context": {
    "current_page": "dashboard",
    "current_operation": "view_results"
  },
  "images": [],
  "files": []
}
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `message` | string | 用户消息 |
| `session_id` | string | 会话 ID（用于历史记录关联） |
| `context` | object | 上下文信息（可选） |
| `images` | array | 图片列表（用于多模态对话） |
| `files` | array | 文件列表 |

**响应** `200 OK`:
```json
{
  "message": "根据分析，您的数据集包含 5 个主要主题，其中最突出的是「机器学习」主题，占比约 35%。",
  "thinking": null
}
```

---

### `GET /api/chat/history/{session_id}` — 获取会话历史

**认证**: 需要

**响应** `200 OK`:
```json
{
  "messages": [
    {"id": 1, "role": "user", "content": "分析一下", "session_id": "default", "created_at": "2026-04-25T10:00:00Z"},
    {"id": 2, "role": "ai", "content": "好的，我来帮您分析...", "session_id": "default", "created_at": "2026-04-25T10:00:01Z"}
  ]
}
```

---

## 7. 结果解读模块 `/api/interpret`

### `POST /api/interpret/metrics` — 解读评估指标

**认证**: 需要

**请求体**:
```json
{
  "job_id": "1",
  "language": "zh",
  "use_llm": true
}
```

**响应** `200 OK`:
```json
{
  "success": true,
  "message": "指标解读完成",
  "data": {"job_id": "1", "interpretation": null}
}
```

> ⚠️ 目前为占位符接口，LLM 解读功能暂未实现。

---

### `POST /api/interpret/topics` — 解读主题语义

**认证**: 需要（同 `/interpret/metrics`）

---

### `POST /api/interpret/summary` — 生成分析摘要

**认证**: 需要（同 `/interpret/metrics`）

---

## 8. 图表分析模块 `/api/vision`

### `POST /api/vision/analyze-chart` — 使用 Qwen-VL 分析图表

**认证**: 需要

**请求体**:
```json
{
  "job_id": "1",
  "chart_name": "topic_distribution.png",
  "chart_url": "https://theta-prod-20260123.oss-cn-shanghai.aliyuncs.com/...",
  "analysis_type": "topic-word",
  "language": "zh"
}
```

**响应** `200 OK`:
```json
{
  "success": true,
  "message": "分析完成",
  "data": {
    "analysis": "此主题主要涉及产品评价和质量相关词汇，核心关键词包括「性能」「质量」「价格」等，表明用户在讨论中关注产品性价比。"
  }
}
```

---

## 错误码说明

| HTTP 状态码 | 说明 |
|-------------|------|
| `400` | 请求参数错误（如必填字段缺失、格式错误） |
| `401` | 未认证（未提供 token 或 token 无效/过期） |
| `403` | 无权限（如访问他人资源） |
| `404` | 资源不存在（如数据集、文件、训练任务） |
| `500` | 服务器内部错误（如后端服务异常、DLC 提交失败） |

---

## 环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NEXT_PUBLIC_API_URL` | 前端调用的后端地址 | `http://localhost:8000` |
| `CLOUD_PROVIDER` | 当前云服务商 | `aliyun` |
| `DATABASE_URL` | 数据库连接字符串 | `postgresql://...` |
| `ALIBABA_CLOUD_ACCESS_KEY_ID` | 阿里云 AccessKey | — |
| `OSS_BUCKET_NAME` | OSS Bucket 名称 | `theta-prod-20260123` |