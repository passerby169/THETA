"""
阿里云 (Aliyun) 云服务商实现
实现 CloudProviderBase 接口，整合 OSS 存储和 PAI-DLC 训练服务
"""

import os
import re
import uuid
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import oss2
from dotenv import load_dotenv

from services.abc.base import CloudProviderBase

load_dotenv()


def _resolve_env_vars(value: Any, env: Dict[str, str] = None) -> Any:
    """
    递归替换配置值中的 ${VAR} 为环境变量
    支持带默认值的语法：${VAR:-default}
    """
    if env is None:
        env = dict(os.environ)

    if isinstance(value, str):
        # 匹配 ${VAR} 或 ${VAR:-default}
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
    else:
        return value


def _load_cloud_config() -> dict:
    """加载云服务商配置（支持 ${VAR} 环境变量引用）"""
    config_path = Path(__file__).parent.parent.parent / "config" / "cloud_providers.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Cloud config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return _resolve_env_vars(config)


# =============================================================================
# AliyunProvider 实现
# =============================================================================

class AliyunProvider(CloudProviderBase):
    """阿里云服务商实现"""

    def __init__(self, config: dict):
        """
        初始化阿里云 provider

        Args:
            config: cloud_providers.yaml 中 providers.aliyun 部分的配置
        """
        self._cfg = config
        self._oss_auth = None
        self._oss_bucket = None
        self._dlc_client = None

    # --------------------------------------------------------------------------
    # 基础信息
    # --------------------------------------------------------------------------

    def get_provider_name(self) -> str:
        return self._cfg.get("provider_name", "aliyun")

    def is_enabled(self) -> bool:
        return self._cfg.get("enabled", True)

    # --------------------------------------------------------------------------
    # OSS 对象存储
    # --------------------------------------------------------------------------

    def _get_oss_auth(self):
        if self._oss_auth is None:
            creds = self._cfg.get("credentials", {})
            access_key_id = creds.get("access_key_id", os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID", ""))
            access_key_secret = creds.get("access_key_secret", os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", ""))
            self._oss_auth = oss2.Auth(access_key_id, access_key_secret)
        return self._oss_auth

    def get_oss_client(self) -> oss2.Bucket:
        """获取 OSS Bucket 客户端"""
        if self._oss_bucket is None:
            oss_cfg = self._cfg.get("oss", {})
            bucket_name = oss_cfg.get("bucket", os.getenv("OSS_BUCKET_NAME", ""))
            endpoint = oss_cfg.get("endpoint", os.getenv("OSS_ENDPOINT", ""))
            self._oss_bucket = oss2.Bucket(self._get_oss_auth(), endpoint, bucket_name)
        return self._oss_bucket

    def _get_internal_endpoint(self) -> str:
        """获取容器内网 OSS endpoint（节省流量费用）"""
        oss_cfg = self._cfg.get("oss", {})
        return oss_cfg.get("internal_endpoint", os.getenv("OSS_ENDPOINT_INTERNAL", ""))

    def _get_oss_path(self, key: str) -> str:
        """获取 OSS 对象完整路径"""
        oss_cfg = self._cfg.get("oss", {})
        return oss_cfg.get("paths", {}).get(key, key)

    def get_object_url(self, oss_key: str, expires_seconds: int = 3600) -> str:
        """获取 OSS 对象的签名 URL"""
        bucket = self.get_oss_client()
        return bucket.sign_url("GET", oss_key, expires_seconds)

    def upload_file(self, local_path: str, oss_key: str) -> str:
        """上传单个文件到 OSS"""
        bucket = self.get_oss_client()
        oss2.resumable_upload(bucket, oss_key, local_path)
        endpoint = self._cfg.get("oss", {}).get("endpoint", "")
        bucket_name = self._cfg.get("oss", {}).get("bucket", "")
        return f"https://{bucket_name}.{endpoint}/{oss_key}"

    def download_file(self, oss_key: str, local_path: str) -> str:
        """从 OSS 下载文件到本地"""
        bucket = self.get_oss_client()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        bucket.get_object_to_file(oss_key, local_path)
        return local_path

    def sync_directory(
        self,
        local_dir: str,
        oss_prefix: str,
        exclude_patterns: Optional[List[str]] = None
    ) -> int:
        """同步本地目录到 OSS"""
        if exclude_patterns is None:
            exclude_patterns = [
                "__pycache__", ".git", ".pyc", ".pyo",
                ".egg-info", ".DS_Store", "*.log", "*.tmp"
            ]

        bucket = self.get_oss_client()
        local_path = Path(local_dir)

        if not local_path.exists():
            raise FileNotFoundError(f"Directory not found: {local_dir}")

        uploaded_count = 0
        for file_path in local_path.rglob("*"):
            if file_path.is_dir():
                continue

            relative_path = file_path.relative_to(local_path)
            if any(pat in str(relative_path) for pat in exclude_patterns):
                continue

            oss_key = f"{oss_prefix.rstrip('/')}/{relative_path}"
            try:
                with open(file_path, "rb") as f:
                    bucket.put_object(oss_key, f)
                uploaded_count += 1
            except Exception as e:
                print(f"[WARN] Failed to upload {file_path}: {e}")

        print(f"[INFO] Synced {uploaded_count} files to oss://{self._cfg.get('oss', {}).get('bucket', '')}/{oss_prefix}")
        return uploaded_count

    def list_objects(self, prefix: str) -> List[str]:
        """列出 OSS 中指定前缀下的所有对象"""
        bucket = self.get_oss_client()
        objects = []
        for obj in oss2.ObjectIterator(bucket, prefix=prefix):
            objects.append(obj.key)
        return objects

    def delete_object(self, oss_key: str) -> None:
        """删除指定 OSS 对象"""
        bucket = self.get_oss_client()
        bucket.delete_object(oss_key)

    def delete_prefix(self, prefix: str) -> int:
        """删除指定前缀下的所有 OSS 对象"""
        bucket = self.get_oss_client()
        count = 0
        for obj in oss2.ObjectIterator(bucket, prefix=prefix):
            bucket.delete_object(obj.key)
            count += 1
        return count

    # --------------------------------------------------------------------------
    # STS 临时凭证（前端直传）
    # --------------------------------------------------------------------------

    def get_sts_token(self, username: str, dataset_name: str) -> Dict[str, Any]:
        """获取 STS 临时凭证（用于前端直传 OSS）"""
        from utils.sts_util import get_sts_token as _get_sts_token
        return _get_sts_token(username, dataset_name)

    # --------------------------------------------------------------------------
    # PAI-DLC 训练任务
    # --------------------------------------------------------------------------

    def _get_dlc_client(self):
        """获取 DLC 客户端"""
        if self._dlc_client is None:
            from alibabacloud_pai_dlc20201203.client import Client as DLCClient
            from alibabacloud_tea_openapi import models as open_api_models

            creds = self._cfg.get("credentials", {})
            region = self._cfg.get("region", "cn-shanghai")

            config = open_api_models.Config(
                access_key_id=creds.get("access_key_id", os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "")),
                access_key_secret=creds.get("access_key_secret", os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")),
                region_id=region
            )
            config.endpoint = f"pai-dlc.{region}.aliyuncs.com"
            self._dlc_client = DLCClient(config)
        return self._dlc_client

    def _get_data_sources(self, username: str) -> List:
        """配置 OSS 数据源挂载（DLC 容器内访问 OSS）"""
        from alibabacloud_pai_dlc20201203 import models as dlc_models

        oss_cfg = self._cfg.get("oss", {})
        bucket_name = oss_cfg.get("bucket", "")
        internal_ep = self._get_internal_endpoint()

        paths = oss_cfg.get("paths", {})
        data_sources = [
            dlc_models.CreateJobRequestDataSources(
                uri=f"oss://{bucket_name}.{internal_ep}/{paths.get('code', 'code')}",
                mount_path="/mnt/code"
            ),
            dlc_models.CreateJobRequestDataSources(
                uri=f"oss://{bucket_name}.{internal_ep}/{paths.get('models', 'models')}",
                mount_path="/mnt/models"
            ),
            dlc_models.CreateJobRequestDataSources(
                uri=f"oss://{bucket_name}.{internal_ep}/{paths.get('raw_data', 'raw_data')}",
                mount_path="/mnt/raw_data"
            ),
            dlc_models.CreateJobRequestDataSources(
                uri=f"oss://{bucket_name}.{internal_ep}/{paths.get('results', 'results')}",
                mount_path="/mnt/results"
            ),
        ]
        return data_sources

    def _generate_run_id(self) -> str:
        """生成唯一的运行 ID: timestamp_uuid"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{uuid.uuid4().hex[:8]}"

    def submit_training_job(
        self,
        user_id: int,
        username: str,
        file_id: int,
        file_path: str,
        job_id: Optional[int] = None,
        dataset_name: Optional[str] = None,
        model_type: str = "theta",
        model_size: str = "0.6B",
        mode: str = "zero_shot",
        num_topics: int = 20,
        epochs: int = 100,
        language: str = "chinese",
        vocab_size: int = 5000,
        **extra_kwargs
    ) -> Optional[str]:
        """提交 DLC 训练任务"""
        from alibabacloud_pai_dlc20201203 import models as dlc_models

        if not dataset_name:
            dataset_name = f"dataset_{file_id}"

        run_id = self._generate_run_id()
        client = self._get_dlc_client()

        # 输入/输出路径
        raw_data_prefix = self._get_oss_path("raw_data")
        results_prefix = self._get_oss_path("results")
        input_dir = f"/mnt/raw_data/{username}/{dataset_name}"
        output_dir = f"/mnt/results/{username}/{dataset_name}/{model_type}/{run_id}"

        # 环境变量
        env_vars = {
            "INPUT_DIR": input_dir,
            "OUTPUT_DIR": output_dir,
            "WORKSPACE_DIR": f"{output_dir}/workspace",
            "MODEL_DIR": f"{output_dir}/model",
            "EVALUATION_DIR": f"{output_dir}/evaluation",
            "VISUALIZATION_DIR": f"{output_dir}/visualization",
            "PROJECT_ROOT": "/mnt/code",
            "DATA_DIR": "/mnt/raw_data",
            "RESULT_DIR": "/mnt/results",
            "USER_ID": str(user_id),
            "USERNAME": username,
            "DATASET_NAME": dataset_name,
            "RUN_ID": run_id,
            "MODEL_TYPE": model_type,
            "MODEL_SIZE": model_size,
            "MODE": mode,
            "NUM_TOPICS": str(num_topics),
            "EPOCHS": str(epochs),
            "LANGUAGE": language,
            "VOCAB_SIZE": str(vocab_size),
            "JOB_ID": str(job_id) if job_id else run_id,
            "API_BASE_URL": os.getenv("API_BASE_URL", "http://localhost:8000"),
            "SECRET_KEY": os.getenv("SECRET_KEY", ""),
            "PYTHONUNBUFFERED": "1",
            # 传递额外参数
            **{k: str(v) for k, v in extra_kwargs.items()}
        }

        # DLC 算例配置
        dlc_cfg = self._cfg.get("dlc", {})
        ecs_spec = os.getenv("DLC_ECS_SPEC", dlc_cfg.get("default_ecs_spec", "ecs.gn7i-c8g1.2xlarge"))
        dlc_image = os.getenv("DLC_IMAGE", dlc_cfg.get("default_image", ""))

        pod_spec = dlc_models.JobSpec(
            type="Worker",
            image=dlc_image,
            pod_count=1,
            ecs_spec=ecs_spec,
        )

        # 训练命令
        user_command = "python /mnt/code/dlc_entry.py"

        create_job_request = dlc_models.CreateJobRequest(
            workspace_id=self._cfg.get("pai", {}).get("workspace_id", os.getenv("PAI_WORKSPACE_ID", "")),
            display_name=f"theta_{model_type}_{username}_{dataset_name}_{run_id}",
            job_type="PyTorchJob",
            job_specs=[pod_spec],
            data_sources=self._get_data_sources(username),
            user_command=user_command,
            envs=env_vars,
        )

        try:
            response = client.create_job(create_job_request)
            if response and response.body and response.body.job_id:
                return response.body.job_id
            return None
        except Exception as e:
            print(f"Failed to submit DLC job: {e}")
            raise e

    def get_job_status(self, job_id: str) -> Optional[str]:
        """查询 DLC 任务状态"""
        from alibabacloud_pai_dlc20201203 import models as dlc_models

        if not job_id:
            return None

        try:
            client = self._get_dlc_client()
            request = dlc_models.GetJobRequest()
            response = client.get_job(job_id, request)

            if response and response.body:
                dlc_status = response.body.status
                status_mapping = {
                    "Waiting": "pending",
                    "Running": "running",
                    "Succeeded": "succeeded",
                    "Failed": "failed",
                    "Stopped": "failed",
                }
                return status_mapping.get(dlc_status, dlc_status.lower() if dlc_status else None)
            return None
        except Exception as e:
            print(f"Failed to get DLC job status: {e}")
            return None

    def stop_job(self, job_id: str) -> bool:
        """停止 DLC 训练任务"""
        try:
            client = self._get_dlc_client()
            request = None
            client.stop_job(job_id, request)
            return True
        except Exception:
            return False

    def get_dlc_config(self) -> Dict[str, Any]:
        """获取 DLC 配置信息（用于前端展示）"""
        dlc_cfg = self._cfg.get("dlc", {})
        return {
            "available_ecs_specs": dlc_cfg.get("available_ecs_specs", []),
            "default_ecs_spec": dlc_cfg.get("default_ecs_spec", "ecs.gn7i-c8g1.2xlarge"),
            "default_image": dlc_cfg.get("default_image", ""),
            "provider": "aliyun",
        }

    # --------------------------------------------------------------------------
    # 健康检查
    # --------------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        result = {
            "provider": self.get_provider_name(),
            "status": "ok",
            "message": "",
            "details": {}
        }

        # 检查 OSS 连接
        try:
            bucket = self.get_oss_client()
            bucket.get_bucket_info()
            result["details"]["oss"] = {"status": "ok", "bucket": self._cfg.get("oss", {}).get("bucket", "")}
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"OSS connection failed: {e}"
            result["details"]["oss"] = {"status": "error", "error": str(e)}

        # 检查 DLC 连接
        try:
            client = self._get_dlc_client()
            result["details"]["dlc"] = {"status": "ok", "workspace_id": self._cfg.get("pai", {}).get("workspace_id", "")}
        except Exception as e:
            result["details"]["dlc"] = {"status": "warning", "error": str(e)}

        return result