# THETA DLC 云训练部署包

`dlc_deployment/` 只保留 THETA 接入云训练需要的内容，不再复制 THETA 开源核心代码。模型、清洗、预处理、训练和可视化的正式实现以 `theta_project/THETA/` 为准；这里负责把这些能力接到阿里云 PAI-DLC。

## 保留内容

| 文件 | 用途 |
|------|------|
| `dlc_entry.py` | DLC 容器入口脚本，负责依赖检查、读取环境变量、数据清洗、预处理、训练、可视化、日志和回调。 |
| `README.md` / `README_zh.md` | 云训练部署说明。 |

已移除重复内容：

| 内容 | 原因 |
|------|------|
| `services/dlc_service.py` | 后端已有 `theta_project/services/dlc_service.py` 和 provider 实现，保留两份会造成参数和镜像配置不一致。 |
| `mkdocs.yml` | 这是 THETA 文档站配置，和 DLC 部署包无关。 |
| 大段 THETA README | 与 `theta_project/THETA/` 重复，容易过期。 |

## OSS 目录约定

提交 DLC 任务前，后端会把运行所需文件同步到 OSS：

```
oss://<bucket>/code/dlc_entry.py
oss://<bucket>/code/src/models/       # 来自 theta_project/THETA/src/models/
oss://<bucket>/code/scripts/          # 来自 theta_project/THETA/scripts/
oss://<bucket>/code/services/         # 后端兼容 helper
oss://<bucket>/models/                # Qwen、SBERT 等模型
oss://<bucket>/raw_data/              # 用户上传数据
oss://<bucket>/results/               # 训练输出
```

DLC 容器挂载后对应为：

```
/mnt/code
/mnt/models
/mnt/raw_data
/mnt/results
```

容器启动命令固定为：

```bash
python /mnt/code/dlc_entry.py
```

## 后端对接点

| 文件 | 作用 |
|------|------|
| `theta_project/services/dlc_service.py` | 兼容旧调用的 DLC 提交与状态查询。 |
| `theta_project/services/providers/aliyun.py` | 云服务商抽象下的阿里云 OSS + PAI-DLC 实现。 |
| `theta_project/utils/oss_util.py` | 将 `dlc_entry.py` 和当前 THETA 代码同步到 OSS。 |
| `theta_project/config/cloud_providers.yaml` | 区域、bucket、挂载路径、默认镜像、默认规格等配置。 |

## 发布流程

1. 确认 `theta_project/THETA/` 是当前要发布的 THETA 核心代码。
2. 调整 `theta_project/config/cloud_providers.yaml` 或环境变量中的 OSS、DLC 镜像、资源规格。
3. 调用 `sync_theta_project_to_oss()`，同步 `dlc_entry.py`、`THETA/src/models/`、`THETA/scripts/` 和后端 helper。
4. 后端调用 `submit_job()` 提交 PAI-DLC 任务。
5. DLC 运行完成后，结果写入 `/mnt/results/{username}/{dataset}/`，并回调后端 `/api/train/callback`。

## 关键环境变量

| 变量 | 说明 |
|------|------|
| `USERNAME` / `THETA_USER_ID` | 用户标识。 |
| `DATASET_NAME` / `THETA_DATASET` | 数据集名称。 |
| `MODEL_TYPE` | 模型类型，默认 `theta`。 |
| `MODEL_SIZE` | Qwen 模型规格，如 `0.6B`、`4B`、`8B`。 |
| `NUM_TOPICS` | 主题数量。 |
| `EPOCHS` | 训练轮数。 |
| `LANGUAGE` | `chinese` 或 `english`。 |
| `API_BASE_URL` | 训练完成回调的后端地址。 |
| `SECRET_KEY` | 回调校验密钥。 |
| `DLC_ECS_SPEC` | DLC 资源规格。 |
| `DLC_IMAGE` | DLC 运行镜像。 |

## 注意事项

- 不要在 `dlc_deployment/` 再放一份 THETA 源码；DLC 运行时从 `/mnt/code/src/models/` 加载同步后的核心代码。
- 修改训练流程时优先改 `dlc_entry.py`；修改模型算法时优先改 `theta_project/THETA/src/models/`。
- 每次发布前重新同步 OSS，避免 DLC 使用旧入口脚本或旧模型代码。
