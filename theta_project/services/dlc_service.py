"""
DLC 训练任务服务 — 支持多云服务商配置

配置读取顺序（优先级从高到低）:
  1. 环境变量（最高优先级，允许运行时覆盖）
  2. config/cloud_providers.yaml 中的 active_provider 配置
  3. 代码默认值

切换云服务商：修改 config/cloud_providers.yaml 中的 active_provider 字段，
或设置 CLOUD_PROVIDER 环境变量。

支持的云服务商: aliyun（默认）
"""

import os
import re
import uuid
from pathlib import Path
from typing import Optional
from alibabacloud_pai_dlc20201203.client import Client as DLCClient
from alibabacloud_pai_dlc20201203 import models as dlc_models
from alibabacloud_tea_openapi import models as open_api_models
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 云配置加载（支持 YAML + 环境变量插值）
# ---------------------------------------------------------------------------

def _resolve_env_var(value: str) -> str:
    """将 ${ENV_VAR} 或 ${ENV_VAR:-default} 替换为实际环境变量值。"""
    def _replace(match: re.Match) -> str:
        expr = match.group(1)
        if ":-" in expr:
            name, default = expr.split(":-", 1)
            return os.getenv(name.strip(), default.strip())
        return os.getenv(expr.strip(), "")
    return re.sub(r"\$\{([^}]+)\}", _replace, value)


def _deep_resolve(obj):
    """递归解析嵌套 dict/list 中的 ${ENV_VAR} 占位符。"""
    if isinstance(obj, str):
        return _resolve_env_var(obj)
    if isinstance(obj, dict):
        return {k: _deep_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_resolve(v) for v in obj]
    return obj


def _load_yaml_config() -> Optional[dict]:
    """
    加载 config/cloud_providers.yaml 并解析 ${ENV_VAR} 占位符。
    加载失败时返回 None（回退到纯环境变量模式）。
    """
    project_root = Path(__file__).resolve().parents[2]  # services/ → theta_project/ → root
    yaml_path = project_root / "config" / "cloud_providers.yaml"
    if not yaml_path.exists():
        return None
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return _deep_resolve(raw)
    except Exception:
        return None


_cloud_config = _load_yaml_config()
_active_provider = os.getenv("CLOUD_PROVIDER") or (
    _cloud_config.get("active_provider") if _cloud_config else None
) or "aliyun"


def _provider_config() -> dict:
    """返回当前激活云服务商的配置字典（已解析 ${ENV_VAR}）。"""
    if _cloud_config and _active_provider in _cloud_config.get("providers", {}):
        return _cloud_config["providers"][_active_provider]
    return {}


def _cfg(key: str, env_var: str, default: str = "") -> str:
    """
    读取配置键，优先级:
      1. 环境变量
      2. cloud_providers.yaml active_provider.{key}
      3. default
    """
    env_val = os.getenv(env_var)
    if env_val is not None and env_val.strip():
        return env_val.strip()
    yaml_val = _provider_config().get(key)
    if yaml_val and str(yaml_val).strip():
        return str(yaml_val).strip()
    return default


# ---------------------------------------------------------------------------
# 云服务商凭证
# ---------------------------------------------------------------------------

ALIBABA_CLOUD_ACCESS_KEY_ID = _cfg("credentials.access_key_id", "ALIBABA_CLOUD_ACCESS_KEY_ID")
ALIBABA_CLOUD_ACCESS_KEY_SECRET = _cfg("credentials.access_key_secret", "ALIBABA_CLOUD_ACCESS_KEY_SECRET")
ALIBABA_CLOUD_REGION_ID = _cfg("region", "ALIBABA_CLOUD_REGION_ID", "cn-shanghai")
PAI_WORKSPACE_ID = _cfg("pai.workspace_id", "PAI_WORKSPACE_ID", "")
OSS_BUCKET_NAME = _cfg("oss.bucket", "OSS_BUCKET_NAME", "")
OSS_ENDPOINT = _cfg("oss.endpoint", "OSS_ENDPOINT", "")
API_BASE_URL = os.getenv("API_BASE_URL", "")

# DLC 训练参数默认值（可被 YAML 和环境变量覆盖）
DLC_ECS_SPEC = os.getenv("DLC_ECS_SPEC") or _provider_config().get("dlc", {}).get("default_ecs_spec", "ecs.gn7i-c8g1.2xlarge")
DLC_IMAGE = os.getenv("DLC_IMAGE") or _provider_config().get("dlc", {}).get("default_image", "dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai/pytorch-training:2.7-gpu-py312-cu128-ubuntu24.04")

# ---------------------------------------------------------------------------
# OSS 内部端点（容器内访问用，节省流量费用）
# ---------------------------------------------------------------------------

def _oss_internal_endpoint(public_endpoint: str) -> str:
    """将公共 OSS endpoint 转换为内网 endpoint。"""
    # 阿里云: oss-cn-shanghai.aliyuncs.com → oss-cn-shanghai-internal.aliyuncs.com
    if "aliyuncs.com" in public_endpoint and "-internal." not in public_endpoint:
        return public_endpoint.replace(".aliyuncs.com", "-internal.aliyuncs.com")
    return public_endpoint


# ---------------------------------------------------------------------------
# 用户隔离路径（多用户 OSS 目录结构）
# ---------------------------------------------------------------------------

def user_raw_data_path(username: str, dataset_name: str) -> str:
    """OSS raw_data/{username}/{dataset_name}/"""
    return f"raw_data/{username}/{dataset_name}"


def user_results_path(username: str, dataset_name: str, model_type: str, run_id: str) -> str:
    """OSS results/{username}/{dataset_name}/{model_type}/{run_id}/"""
    return f"results/{username}/{dataset_name}/{model_type}/{run_id}"


# ---------------------------------------------------------------------------
# DLC Client
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """生成唯一运行 ID: timestamp_uuid"""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def get_dlc_client() -> DLCClient:
    """根据当前云配置创建 DLC 客户端。"""
    provider = _provider_config()
    dlc_endpoint = provider.get("pai", {}).get("dlc_endpoint", f"pai-dlc.{ALIBABA_CLOUD_REGION_ID}.aliyuncs.com")

    config = open_api_models.Config(
        access_key_id=ALIBABA_CLOUD_ACCESS_KEY_ID,
        access_key_secret=ALIBABA_CLOUD_ACCESS_KEY_SECRET,
        region_id=ALIBABA_CLOUD_REGION_ID,
    )
    config.endpoint = dlc_endpoint
    return DLCClient(config)


# ---------------------------------------------------------------------------
# OSS 数据源挂载
# ---------------------------------------------------------------------------

def get_data_sources(username: str = None) -> list[dlc_models.CreateJobRequestDataSources]:
    """
    配置 DLC 容器的 OSS 数据源挂载。

    挂载结构:
        /mnt/code/      <- oss://bucket/code/         (THETA 训练代码)
        /mnt/models/    <- oss://bucket/models/       (qwen-0.6b/, sbert/)
        /mnt/raw_data/  <- oss://bucket/raw_data/    (用户上传原始数据)
        /mnt/results/   <- oss://bucket/results/     (训练结果输出)

    所有路径使用 OSS 内网 endpoint，节省流量费用。
    """
    provider = _provider_config()
    oss_cfg = provider.get("oss", {})
    public_ep = oss_cfg.get("endpoint", OSS_ENDPOINT) or OSS_ENDPOINT
    internal_ep = oss_cfg.get("internal_endpoint") or _oss_internal_endpoint(public_ep)

    bucket_uri = lambda key: f"oss://{OSS_BUCKET_NAME}.{internal_ep}/{key}" if internal_ep != public_ep \
                              else f"oss://{OSS_BUCKET_NAME}/{key}"

    paths = oss_cfg.get("paths", {}) or {}

    return [
        dlc_models.CreateJobRequestDataSources(
            uri=bucket_uri(paths.get("code", "code/")),
            mount_path="/mnt/code",
        ),
        dlc_models.CreateJobRequestDataSources(
            uri=bucket_uri(paths.get("models", "models/")),
            mount_path="/mnt/models",
        ),
        dlc_models.CreateJobRequestDataSources(
            uri=bucket_uri(paths.get("raw_data", "raw_data/")),
            mount_path="/mnt/raw_data",
        ),
        dlc_models.CreateJobRequestDataSources(
            uri=bucket_uri(paths.get("results", "results/")),
            mount_path="/mnt/results",
        ),
    ]


# ---------------------------------------------------------------------------
# 提交训练任务
# ---------------------------------------------------------------------------

def submit_job(
    user_id: int,
    username: str,
    file_id: int,
    file_path: str,
    job_id: int = None,
    dataset_name: str = None,
    model_type: str = "theta",
    model_size: str = "0.6B",
    mode: str = "zero_shot",
    num_topics: int = 20,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.002,
    hidden_dim: int = 512,
    patience: int = 10,
    language: str = "chinese",
    vocab_size: int = 5000,
) -> tuple[str, str]:
    """
    提交 DLC 训练任务。

    多用户隔离: 所有 OSS 路径以 {username} 为前缀，确保用户间数据完全隔离。
    run_id 由本函数生成，写入 DLC 容器环境变量 TRAINING_RUN_ID，
    DLC 容器将产物写入 /mnt/results/{username}/{dataset}/{model}/{run_id}/。
    回调时将 run_id 回传后端，填入 TrainingJob.run_id 字段。

    Returns
    -------
    (dlc_job_id, run_id)
    """
    client = get_dlc_client()

    if not dataset_name:
        dataset_name = f"dataset_{file_id}"

    run_id = generate_run_id()

    # OSS 路径（挂载后的容器内路径与 OSS 路径前缀相同）
    raw_data_rel = user_raw_data_path(username, dataset_name)  # raw_data/{user}/{dataset}
    results_rel  = user_results_path(username, dataset_name, model_type, run_id)

    input_dir   = f"/mnt/raw_data"       # 容器内 raw_data 挂载点（不含 user 前缀）
    output_dir  = f"/mnt/results/{results_rel}"

    env_vars = {
        # 路径（容器内，与 OSS 挂载点一一对应）
        "INPUT_DIR":        input_dir,
        "RAW_DATA_USER_DIR": f"/mnt/raw_data/{username}/{dataset_name}",
        "OUTPUT_DIR":       output_dir,
        "WORKSPACE_DIR":    f"{output_dir}/workspace",
        "MODEL_DIR":        f"{output_dir}/model",
        "EVALUATION_DIR":  f"{output_dir}/evaluation",
        "VISUALIZATION_DIR":f"{output_dir}/visualization",
        # 兼容旧路径别名
        "PROJECT_ROOT":     "/mnt/code",
        "DATA_DIR":         "/mnt/raw_data",
        "RESULT_DIR":       "/mnt/results",
        # 用户和任务信息
        "THETA_USER_ID":    username,
        "THETA_DATASET":    dataset_name,
        "THETA_MODE":       mode,
        "THETA_JOB_ID":     str(job_id) if job_id else run_id,
        "USER_ID":          str(user_id),
        "USERNAME":         username,
        "DATASET_NAME":     dataset_name,
        "RUN_ID":           run_id,
        "TIMESTAMP":        run_id,
        # 训练参数
        "MODEL_TYPE":       model_type,
        "MODEL_SIZE":       model_size,
        "MODE":             mode,
        "NUM_TOPICS":       str(num_topics),
        "EPOCHS":           str(epochs),
        "BATCH_SIZE":       str(batch_size),
        "LEARNING_RATE":   str(learning_rate),
        "HIDDEN_DIM":       str(hidden_dim),
        "PATIENCE":         str(patience),
        "LANGUAGE":         language,
        "VOCAB_SIZE":       str(vocab_size),
        "JOB_ID":           str(job_id) if job_id else "",
        # 回调
        "API_BASE_URL":     API_BASE_URL or "",
        "SECRET_KEY":       os.getenv("SECRET_KEY", ""),
        "PYTHONUNBUFFERED": "1",
    }

    pod_spec = dlc_models.JobSpec(
        type="Worker",
        image=DLC_IMAGE,
        pod_count=1,
        ecs_spec=DLC_ECS_SPEC,
    )

    create_job_request = dlc_models.CreateJobRequest(
        workspace_id=PAI_WORKSPACE_ID,
        display_name=f"theta_{model_type}_{username}_{dataset_name}_{run_id}",
        job_type="PyTorchJob",
        job_specs=[pod_spec],
        data_sources=get_data_sources(username),
        user_command="python /mnt/code/dlc_entry.py",
        envs=env_vars,
    )

    try:
        response = client.create_job(create_job_request)
        if response and response.body and response.body.job_id:
            return response.body.job_id, run_id
        return None, run_id
    except Exception as e:
        print(f"Failed to submit DLC job: {e}")
        raise


# ---------------------------------------------------------------------------
# 查询任务状态
# ---------------------------------------------------------------------------

def get_job_status(dlc_job_id: str) -> Optional[str]:
    """
    查询 DLC 任务实时状态。

    Returns
    -------
    "pending" | "running" | "succeeded" | "failed" | None
    """
    if not dlc_job_id:
        return None
    try:
        client = get_dlc_client()
        request = dlc_models.GetJobRequest()
        response = client.get_job(dlc_job_id, request)
        if not response or not response.body:
            return None
        dlc_status = response.body.status
        mapping = {
            "Waiting":   "pending",
            "Running":   "running",
            "Succeeded": "succeeded",
            "Failed":    "failed",
            "Stopped":   "failed",
        }
        return mapping.get(dlc_status, dlc_status.lower() if dlc_status else None)
    except Exception as e:
        print(f"Failed to get DLC job status: {e}")
        return None
