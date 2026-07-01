"""
云服务商工厂函数
根据配置动态加载并返回指定云服务商实例
"""

import os
import re
import yaml
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from services.abc.base import CloudProviderBase

load_dotenv()

# 全局 provider 实例缓存
_provider_instance: Optional[CloudProviderBase] = None


def _resolve_env_vars(value, env: dict = None) -> str:
    """递归替换配置值中的 ${VAR} 为环境变量"""
    if env is None:
        env = dict(os.environ)

    if isinstance(value, str):
        pattern = r'\$\{([^}:]+)(:-([^}]*))?\}'
        matches = re.findall(pattern, value)
        for full_match, var_name, _, default_val in matches:
            resolved = env.get(var_name, default_val if default_val is not None else "")
            value = value.replace(full_match, resolved)
        return value
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v, env) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item, env) for item in value]
    return value


def _load_cloud_config() -> dict:
    """加载云服务商配置文件"""
    config_path = Path(__file__).parent.parent / "config" / "cloud_providers.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Cloud config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return _resolve_env_vars(config)


def get_cloud_provider(provider_name: str = None) -> CloudProviderBase:
    """
    获取云服务商实例（工厂函数）

    优先级：
        1. 传入的 provider_name 参数
        2. 环境变量 CLOUD_PROVIDER
        3. cloud_providers.yaml 中的 active_provider 配置

    Args:
        provider_name: 云服务商名称（如 'aliyun' / 'aws' / 'tencent'）

    Returns:
        CloudProviderBase 实例

    Raises:
        ValueError: 当指定的云服务商不存在或未启用时
    """
    global _provider_instance

    config = _load_cloud_config()

    # 确定要使用的云服务商
    active = (
        provider_name
        or os.getenv("CLOUD_PROVIDER")
        or config.get("active_provider", "aliyun")
    )

    # 如果已有缓存且是同一云服务商，直接返回
    if _provider_instance is not None and _provider_instance.get_provider_name() == active:
        return _provider_instance

    providers = config.get("providers", {})

    if active not in providers:
        available = list(providers.keys())
        raise ValueError(
            f"Cloud provider '{active}' not found in config. "
            f"Available providers: {available}"
        )

    provider_cfg = providers[active]

    # 检查是否启用
    if not provider_cfg.get("enabled", False):
        raise ValueError(
            f"Cloud provider '{active}' is disabled. "
            f"Set enabled: true in config/cloud_providers.yaml to enable it."
        )

    # 动态加载 provider 实现
    if active == "aliyun":
        from services.providers.aliyun import AliyunProvider
        _provider_instance = AliyunProvider(provider_cfg)
    elif active == "aws":
        # TODO: 实现 AWS provider
        raise NotImplementedError("AWS provider is not yet implemented")
    elif active == "tencent":
        # TODO: 实现腾讯云 provider
        raise NotImplementedError("Tencent provider is not yet implemented")
    elif active == "huawei":
        # TODO: 实现华为云 provider
        raise NotImplementedError("Huawei provider is not yet implemented")
    else:
        raise ValueError(f"Unknown cloud provider: {active}")

    return _provider_instance


def reset_cloud_provider() -> None:
    """重置缓存的 provider 实例（用于测试或配置切换）"""
    global _provider_instance
    _provider_instance = None


# =============================================================================
# 兼容性别名（保留旧代码的调用方式）
# =============================================================================

def get_oss_bucket():
    """
    兼容旧代码：获取 OSS bucket 客户端
    旧代码: from utils.oss_util import get_oss_bucket
    新代码: from services.cloud_provider_factory import get_cloud_provider; provider.get_oss_client()
    """
    return get_cloud_provider().get_oss_client()


def submit_job(*args, **kwargs):
    """兼容旧代码：提交 DLC 训练任务"""
    return get_cloud_provider().submit_training_job(*args, **kwargs)


def get_job_status(job_id: str) -> Optional[str]:
    """兼容旧代码：查询 DLC 任务状态"""
    return get_cloud_provider().get_job_status(job_id)


def sync_code_to_oss():
    """兼容旧代码：同步代码到 OSS"""
    from utils.oss_util import sync_code_to_oss as _sync
    return _sync()


def get_oss_file_url(username: str, dataset_name: str, filename: str) -> str:
    """兼容旧代码：获取 OSS 文件路径"""
    from utils.sts_util import get_oss_file_url as _get_oss_file_url
    return _get_oss_file_url(username, dataset_name, filename)


def get_sts_token(username: str, dataset_name: str) -> dict:
    """兼容旧代码：获取 STS 临时凭证"""
    return get_cloud_provider().get_sts_token(username, dataset_name)


def generate_upload_policy(username: str, dataset_name: str, max_size_mb: int = 500) -> dict:
    """兼容旧代码：生成上传策略"""
    from utils.sts_util import generate_upload_policy as _gen_policy
    return _gen_policy(username, dataset_name, max_size_mb)


def get_oss_config() -> dict:
    """获取当前云服务商的 OSS 配置"""
    provider = get_cloud_provider()
    return provider._cfg.get("oss", {})


def get_dlc_config() -> dict:
    """获取当前云服务商的 DLC 配置"""
    return get_cloud_provider().get_dlc_config()


def health_check() -> dict:
    """健康检查"""
    return get_cloud_provider().health_check()