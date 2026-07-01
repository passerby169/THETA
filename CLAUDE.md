# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a monorepo containing two main components:

| Directory | Technology | Description |
|-----------|-----------|-------------|
| `theta_project/` | Python / FastAPI | THETA topic modeling ML platform + backend API |
| `theta.code-soul.com/` | Next.js 16 / TypeScript | Frontend web application |

---

## `theta_project/` — Python ML Platform

### Key Sub-directories

- `THETA/` — Main open-source THETA source code (topic modeling research)
- `dlc_deployment/` — PAI-DLC container entrypoint and cloud training deployment notes
- `main.py` — FastAPI backend server (user auth, file upload, training job submission, WebSocket)
- `services/dlc_service.py` — DLC (Deep Learning Container) job submission for GPU training
- `app/` — SQLAlchemy database models (User, File, TrainingJob, ChatMessage)
- `utils/` — Alibaba Cloud OSS utilities, STS token generation

### Quick Start (Research / CLI)

```bash
# 1. Environment setup (install deps + download models)
cd theta_project/THETA
bash scripts/env_setup.sh
source scripts/env_setup.sh

# 2. One-click pipeline (clean → preprocess → train → evaluate → visualize)
bash scripts/quick_start.sh my_dataset --language zh   # Chinese
bash scripts/quick_start.sh my_dataset --language en   # English

# 3. Expert mode: run specific model with full parameter control
python src/models/run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --num_topics 20 \
    --language en

# 4. Data preprocessing only (generates BOW matrix + embeddings)
python src/models/prepare_data.py --dataset my_dataset --model theta --model_size 0.6B
```

### Supported Models

- **THETA**: Qwen embedding-based neural topic model (0.6B / 4B / 8B sizes)
- **Traditional**: LDA, HDP, STM, BTM
- **Neural Baselines**: ETM, CTM, DTM, NVDM, GSM, ProdLDA, BERTopic

### Result Output

| Model Type | Output Path |
|-----------|-------------|
| THETA | `result/{dataset}/{model_size}/theta/exp_{timestamp}/` |
| Baseline | `result/{dataset}/{user_id}/{model}/exp_{timestamp}/` |

### Evaluation Metrics

7 gold-standard metrics: TD, iRBO, NPMI, C_V, UMass, Exclusivity, PPL.

### Backend Server

```bash
# Development
python main.py

# Production (PM2)
pm2 start ecosystem.config.js

# Key endpoints
POST /api/auth/register
POST /api/auth/login
POST /api/datasets/upload
POST /api/training/submit
GET  /api/ws  (WebSocket for training progress)
```

---

## `theta.code-soul.com/` — Next.js Frontend

Frontend commands are defined in `theta.code-soul.com/package.json`. The app uses the Next.js App Router under `theta.code-soul.com/app/`, shared components under `theta.code-soul.com/components/`, and API clients under `theta.code-soul.com/lib/api/`.
