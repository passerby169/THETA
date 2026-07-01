# THETA 协作代码仓库

本仓库用于协作开发 THETA 平台的前端与后端代码。THETA 是一个面向社会科学文本分析的 AI 主题建模系统，支持数据上传、文本清洗、主题模型训练、训练结果可视化、AI 对话分析，以及基于云算力的 DLC 训练任务调度。

仓库当前包含两个主要工程：

- `theta_project/`：Python 后端、THETA 模型接入、数据库、OSS/STS、DLC 云训练调度（目前是以阿里云为开发目标）。
- `theta.code-soul.com/`：Next.js 前端应用，负责用户界面、数据上传、训练任务提交、结果展示和对话交互。

## 目录结构

```text
theta/
├── theta_project/                 # Python 后端与训练服务
│   ├── main.py                    # FastAPI 服务入口
│   ├── app/database.py            # SQLAlchemy 数据模型
│   ├── services/                  # 训练任务、云服务商、DLC 调度
│   ├── services/providers/        # 云服务商实现，当前重点为阿里云
│   ├── utils/                     # OSS、STS、路径、Prompt 等工具
│   ├── config/                    # 云服务和 API 配置
│   ├── docs/                      # 后端 API 文档
│   ├── dlc_deployment/            # DLC 容器入口与云训练部署说明
│   └── THETA/                     # THETA 核心模型代码
│
├── theta.code-soul.com/            # Next.js 前端
│   ├── app/                       # App Router 页面
│   ├── components/                # 页面组件和通用 UI
│   ├── contexts/                  # React Context
│   ├── hooks/                     # 前端 Hooks
│   ├── lib/api/                   # 后端 API 客户端与端点配置
│   └── public/                    # 静态资源
│
├── docker-compose.yml              # 本地/部署编排
├── CLAUDE.md                       # 代码协作辅助说明
└── README.md                       # 仓库说明
```

## 本地启动

### Docker 启动

```bash
docker-compose up -d
```

启动后访问：

- 前端：`http://localhost:3000`
- 后端 API：`http://localhost:8000`
- 后端 API 文档：`http://localhost:8000/docs`

### 本地开发启动

后端：

```bash
cd theta_project
pip install -r requirements.txt
python main.py
```

前端：

```bash
cd theta.code-soul.com
npm install
npm run dev
```

## 核心能力

| 模块 | 说明 |
|------|------|
| 用户与数据 | 用户注册登录、数据集上传、OSS 直传、数据集管理 |
| 文本处理 | 文档格式转换、文本清洗、分词、BOW 与嵌入生成 |
| 模型训练 | 支持 THETA、LDA、BERTopic、CTM、DTM 等主题模型 |
| 云训练 | 通过阿里云 OSS + PAI-DLC 提交 GPU 训练任务 |
| 结果展示 | 主题词、主题分布、评估指标、可视化图表 |
| AI 分析 | 基于训练结果进行对话式解释和分析 |

## 关键文件

| 文件 | 用途 |
|------|------|
| `theta_project/main.py` | 后端 API 服务入口 |
| `theta_project/app/database.py` | 用户、文件、训练任务、聊天记录等数据模型 |
| `theta_project/services/dlc_service.py` | DLC 任务提交与状态查询 |
| `theta_project/services/providers/aliyun.py` | 阿里云 OSS、STS、PAI-DLC provider 实现 |
| `theta_project/utils/oss_util.py` | 本地代码与训练入口同步到 OSS |
| `theta_project/dlc_deployment/dlc_entry.py` | DLC 容器内执行的训练入口脚本 |
| `theta_project/config/cloud_providers.yaml` | 云服务商、OSS、DLC、回调等配置 |
| `theta_project/config/api_endpoints.yaml` | 后端接口配置说明 |
| `theta.code-soul.com/lib/api/backend.ts` | 前端后端 API 客户端 |
| `theta.code-soul.com/lib/api/endpoints-config.ts` | 前端 API 端点常量 |

## 环境配置

后端环境变量放在：

```text
theta_project/.env
```

前端环境变量放在：

```text
theta.code-soul.com/.env.local
```

常用配置包括：

| 配置 | 说明 |
|------|------|
| `DATABASE_URL` | PostgreSQL 或 SQLite 数据库连接 |
| `OSS_BUCKET_NAME` | OSS bucket 名称 |
| `OSS_ENDPOINT` | OSS endpoint |
| `ALIBABA_CLOUD_ACCESS_KEY_ID` | 阿里云 AccessKey ID |
| `ALIBABA_CLOUD_ACCESS_KEY_SECRET` | 阿里云 AccessKey Secret |
| `DLC_ECS_SPEC` | DLC 训练资源规格 |
| `DLC_IMAGE` | DLC 训练镜像 |
| `SECRET_KEY` | 后端鉴权和训练回调校验密钥 |
| `NEXT_PUBLIC_API_URL` | 前端访问后端的 API 地址 |

不要把真实密钥提交到仓库。

## DLC 云训练流程

云训练相关代码分为三部分：

| 位置 | 责任 |
|------|------|
| `theta_project/services/dlc_service.py` | 后端提交 DLC 任务 |
| `theta_project/utils/oss_util.py` | 同步 `dlc_entry.py` 与 THETA 核心模型代码到 OSS |
| `theta_project/dlc_deployment/dlc_entry.py` | DLC 容器启动后执行完整训练流程 |

运行时 OSS 结构约定：

```text
oss://<bucket>/code/dlc_entry.py
oss://<bucket>/code/src/models/
oss://<bucket>/code/scripts/
oss://<bucket>/models/
oss://<bucket>/raw_data/
oss://<bucket>/results/
```

DLC 容器挂载后使用：

```text
/mnt/code
/mnt/models
/mnt/raw_data
/mnt/results
```

容器入口命令：

```bash
python /mnt/code/dlc_entry.py
```

## 开发协作约定

- 前端改动集中在 `theta.code-soul.com/`。
- 后端 API、数据库、云训练、OSS 相关改动集中在 `theta_project/`。
- THETA 模型算法相关改动优先放在 `theta_project/THETA/`。
- DLC 容器流程只改 `theta_project/dlc_deployment/dlc_entry.py`，不要在 `dlc_deployment/` 里复制 THETA 源码。
- 配置优先放到 `.env` 或 `config/*.yaml`，避免把环境差异写死在业务代码里。
- 新增接口时同步更新前端 API 客户端和后端接口文档。

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Next.js 16, React 19, TypeScript, TailwindCSS, shadcn/ui |
| 后端 | Python 3.10+, FastAPI, SQLAlchemy, Pydantic |
| 数据库 | PostgreSQL / SQLite |
| 对象存储 | 阿里云 OSS |
| 云训练 | 阿里云 PAI-DLC |
| 模型 | THETA, LDA, BERTopic, CTM, DTM 等 |
| AI 对话 | DashScope / Qwen 系列模型 |

