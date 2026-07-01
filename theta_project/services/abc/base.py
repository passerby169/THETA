"""
云服务商抽象基类
定义所有云服务商必须实现的统一接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class CloudProviderBase(ABC):
    """云服务商抽象基类 — 所有云服务商必须实现此接口"""

    # --------------------------------------------------------------------------
    # 基础信息
    # --------------------------------------------------------------------------

    @abstractmethod
    def get_provider_name(self) -> str:
        """返回云服务商名称，如 'aliyun' / 'aws' / 'tencent'"""

    @abstractmethod
    def is_enabled(self) -> bool:
        """返回该云服务商是否启用"""

    # --------------------------------------------------------------------------
    # OSS/S3 对象存储
    # --------------------------------------------------------------------------

    @abstractmethod
    def get_oss_client(self) -> Any:
        """获取对象存储客户端（阿里云为 oss2.Bucket，AWS 为 boto3 client）"""

    @abstractmethod
    def upload_file(self, local_path: str, oss_key: str) -> str:
        """
        上传单个文件到对象存储
        Args:
            local_path: 本地文件路径
            oss_key: OSS/S3 对象路径（如 'raw_data/user/dataset/file.csv'）
        Returns:
            对象存储的 URL（如 'https://bucket.endpoint/oss_key'）
        """

    @abstractmethod
    def download_file(self, oss_key: str, local_path: str) -> str:
        """
        从对象存储下载文件到本地
        Args:
            oss_key: OSS/S3 对象路径
            local_path: 本地保存路径
        Returns:
            本地文件路径
        """

    @abstractmethod
    def sync_directory(
        self,
        local_dir: str,
        oss_prefix: str,
        exclude_patterns: Optional[List[str]] = None
    ) -> int:
        """
        同步本地目录到对象存储
        Args:
            local_dir: 本地目录路径
            oss_prefix: OSS/S3 前缀路径（如 'code/scripts/'）
            exclude_patterns: 排除的文件/目录模式列表
        Returns:
            上传的文件数量
        """

    @abstractmethod
    def list_objects(self, prefix: str) -> List[str]:
        """
        列出对象存储中指定前缀下的所有对象路径
        Args:
            prefix: 对象存储路径前缀
        Returns:
            对象路径列表
        """

    @abstractmethod
    def delete_object(self, oss_key: str) -> None:
        """删除指定对象"""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """
        删除指定前缀下的所有对象
        Args:
            prefix: 对象存储路径前缀
        Returns:
            删除的对象数量
        """

    @abstractmethod
    def get_object_url(self, oss_key: str, expires_seconds: int = 3600) -> str:
        """
        获取对象的签名 URL（有时效性的访问链接）
        Args:
            oss_key: OSS/S3 对象路径
            expires_seconds: URL 有效期（秒）
        Returns:
            签名 URL
        """

    # --------------------------------------------------------------------------
    # STS 临时凭证（前端直传用）
    # --------------------------------------------------------------------------

    @abstractmethod
    def get_sts_token(self, username: str, dataset_name: str) -> Dict[str, Any]:
        """
        获取 STS 临时凭证（用于前端直传对象存储）
        Args:
            username: 用户名（用于路径隔离）
            dataset_name: 数据集名称
        Returns:
            包含 credentials 的字典，结构：
            {
                "credentials": {
                    "access_key_id": "...",
                    "access_key_secret": "...",
                    "security_token": "...",
                    "expiration": "..."
                },
                "upload_path": "raw_data/{username}/{dataset_name}/",
                "bucket": "bucket-name",
                "endpoint": "endpoint-url",
                "region": "region-id"
            }
        """

    # --------------------------------------------------------------------------
    # DLC/训练任务（AI 训练调度）
    # --------------------------------------------------------------------------

    @abstractmethod
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
        """
        提交 AI 训练任务到云服务商
        Args:
            user_id: 用户 ID
            username: 用户名（用于路径寻址）
            file_id: 文件 ID
            file_path: OSS/S3 上的文件路径
            job_id: 数据库中的训练任务 ID（用于回调）
            dataset_name: 数据集名称（用于目录隔离）
            model_type: 模型类型（如 'theta' / 'lda' / 'bertopic'）
            model_size: Qwen 模型规格（如 '0.6B' / '4B' / '8B'）
            mode: 嵌入模式（如 'zero_shot' / 'supervised' / 'unsupervised'）
            num_topics: 主题数量
            epochs: 训练轮数
            language: 语言（'chinese' / 'english'）
            vocab_size: 词表大小
            **extra_kwargs: 其他可选参数（如 batch_size / learning_rate 等）
        Returns:
            云服务商返回的任务 ID（如 DLC Job ID / SageMaker Training Job Name）
        """

    @abstractmethod
    def get_job_status(self, job_id: str) -> Optional[str]:
        """
        查询训练任务状态
        Args:
            job_id: 云服务商返回的任务 ID
        Returns:
            任务状态字符串（pending / running / succeeded / failed）
        """

    @abstractmethod
    def stop_job(self, job_id: str) -> bool:
        """
        停止训练任务
        Args:
            job_id: 任务 ID
        Returns:
            是否成功停止
        """

    @abstractmethod
    def get_dlc_config(self) -> Dict[str, Any]:
        """
        获取 DLC/训练配置信息（用于前端展示和配置）
        Returns:
            包含可用算例、镜像等配置的字典：
            {
                "available_ecs_specs": [...],
                "default_ecs_spec": "...",
                "default_image": "...",
                "gpu_options": [...]
            }
        """

    # --------------------------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------------------------

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查（检查凭证是否有效、对象存储是否可达等）
        Returns:
            {"status": "ok" / "error", "message": "...", "details": {...}}
        """