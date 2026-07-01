"""
THETA Path Manager - 集中管理所有路径拼接逻辑

路径架构:
=========

1. 输入区 (Raw Data Layer)
   路径格式: /mnt/raw_data/{user_id}/{dataset_name}/
   - data.csv: 用户直传的原始数据集
   - processed_data.csv: 清洗后的标准化格式文件
   特点: 只读（针对训练任务），作为整个流水线的起点

2. 输出区 (Result Layer)
   路径格式: /mnt/results/{user_id}/{dataset_name}/{model_name}/{timestamp}/
   - ./workspace/: 中间产物 (bow_matrix.npy, vocab.json, embeddings.pt)
   - ./model/: 训练好的模型权重 (model.pth, lda_model.joblib)
   - ./evaluation/: 评估指标 (metrics.json)
   - ./visualization/: 图表和可视化报告 (.png)

使用方式:
=========
    from utils.path_manager import get_canonical_paths
    
    paths = get_canonical_paths(
        user_id="test_user",
        dataset_name="policy_docs",
        model_name="lda",
        timestamp="20260318_120000"
    )
    
    # 访问路径
    paths.input_dir          # /mnt/raw_data/test_user/policy_docs/
    paths.output_dir         # /mnt/results/test_user/policy_docs/lda/20260318_120000/
    paths.workspace_dir      # /mnt/results/test_user/policy_docs/lda/20260318_120000/workspace/
    paths.model_dir          # /mnt/results/test_user/policy_docs/lda/20260318_120000/model/
    paths.evaluation_dir     # /mnt/results/test_user/policy_docs/lda/20260318_120000/evaluation/
    paths.visualization_dir  # /mnt/results/test_user/policy_docs/lda/20260318_120000/visualization/
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# 挂载点根目录 (DLC 容器内)
MOUNT_ROOT = "/mnt"
RAW_DATA_MOUNT = f"{MOUNT_ROOT}/raw_data"
RESULTS_MOUNT = f"{MOUNT_ROOT}/results"
MODELS_MOUNT = f"{MOUNT_ROOT}/models"
CODE_MOUNT = f"{MOUNT_ROOT}/code"


class PathValidationError(Exception):
    """路径验证错误"""
    pass


def validate_user_id(user_id: str) -> str:
    """
    验证并清理 user_id
    - 只允许字母、数字、下划线、连字符
    - 不允许路径遍历字符
    """
    if not user_id:
        raise PathValidationError("user_id cannot be empty")
    
    # 检查路径遍历攻击
    if ".." in user_id or "/" in user_id or "\\" in user_id:
        raise PathValidationError(f"Invalid user_id: {user_id} (path traversal detected)")
    
    # 只允许安全字符
    if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
        raise PathValidationError(f"Invalid user_id: {user_id} (only alphanumeric, underscore, hyphen allowed)")
    
    return user_id


def validate_dataset_name(dataset_name: str) -> str:
    """
    验证并清理 dataset_name
    - 只允许字母、数字、下划线、连字符
    - 不允许路径遍历字符
    """
    if not dataset_name:
        raise PathValidationError("dataset_name cannot be empty")
    
    # 检查路径遍历攻击
    if ".." in dataset_name or "/" in dataset_name or "\\" in dataset_name:
        raise PathValidationError(f"Invalid dataset_name: {dataset_name} (path traversal detected)")
    
    # 只允许安全字符
    if not re.match(r'^[a-zA-Z0-9_-]+$', dataset_name):
        raise PathValidationError(f"Invalid dataset_name: {dataset_name} (only alphanumeric, underscore, hyphen allowed)")
    
    return dataset_name


def validate_model_name(model_name: str) -> str:
    """
    验证并清理 model_name
    - 只允许预定义的模型名称
    """
    valid_models = {'lda', 'etm', 'ctm', 'dtm', 'stm', 'theta', 'baseline'}
    
    if not model_name:
        raise PathValidationError("model_name cannot be empty")
    
    model_name_lower = model_name.lower()
    if model_name_lower not in valid_models:
        raise PathValidationError(f"Invalid model_name: {model_name} (allowed: {valid_models})")
    
    return model_name_lower


@dataclass
class CanonicalPaths:
    """
    规范化路径数据类
    
    所有路径都是 Path 对象，可以直接用于文件操作
    """
    # 基础信息
    user_id: str
    dataset_name: str
    model_name: str
    timestamp: str
    
    # 输入区路径
    input_dir: Path = field(init=False)
    input_data_csv: Path = field(init=False)
    input_processed_csv: Path = field(init=False)
    
    # 输出区路径
    output_dir: Path = field(init=False)
    workspace_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    evaluation_dir: Path = field(init=False)
    visualization_dir: Path = field(init=False)
    
    # 具体文件路径
    bow_matrix_path: Path = field(init=False)
    vocab_path: Path = field(init=False)
    embeddings_path: Path = field(init=False)
    metrics_path: Path = field(init=False)
    
    def __post_init__(self):
        # 输入区
        self.input_dir = Path(RAW_DATA_MOUNT) / self.user_id / self.dataset_name
        self.input_data_csv = self.input_dir / "data.csv"
        self.input_processed_csv = self.input_dir / "processed_data.csv"
        
        # 输出区
        self.output_dir = Path(RESULTS_MOUNT) / self.user_id / self.dataset_name / self.model_name / self.timestamp
        self.workspace_dir = self.output_dir / "workspace"
        self.model_dir = self.output_dir / "model"
        self.evaluation_dir = self.output_dir / "evaluation"
        self.visualization_dir = self.output_dir / "visualization"
        
        # 具体文件
        self.bow_matrix_path = self.workspace_dir / "bow_matrix.npy"
        self.vocab_path = self.workspace_dir / "vocab.json"
        self.embeddings_path = self.workspace_dir / "embeddings.pt"
        self.metrics_path = self.evaluation_dir / "metrics.json"
    
    def ensure_dirs(self):
        """创建所有必要的目录"""
        for dir_path in [self.workspace_dir, self.model_dir, self.evaluation_dir, self.visualization_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def to_env_dict(self) -> dict:
        """
        转换为环境变量字典，用于 DLC 注入
        """
        return {
            "INPUT_DIR": str(self.input_dir),
            "OUTPUT_DIR": str(self.output_dir),
            "WORKSPACE_DIR": str(self.workspace_dir),
            "MODEL_DIR": str(self.model_dir),
            "EVALUATION_DIR": str(self.evaluation_dir),
            "VISUALIZATION_DIR": str(self.visualization_dir),
        }
    
    def __str__(self) -> str:
        return f"""CanonicalPaths:
  User: {self.user_id}
  Dataset: {self.dataset_name}
  Model: {self.model_name}
  Timestamp: {self.timestamp}
  
  Input:
    - input_dir: {self.input_dir}
    - data.csv: {self.input_data_csv}
    - processed_data.csv: {self.input_processed_csv}
  
  Output:
    - output_dir: {self.output_dir}
    - workspace/: {self.workspace_dir}
    - model/: {self.model_dir}
    - evaluation/: {self.evaluation_dir}
    - visualization/: {self.visualization_dir}
"""


def get_canonical_paths(
    user_id: str,
    dataset_name: str,
    model_name: str,
    timestamp: Optional[str] = None,
    validate: bool = True
) -> CanonicalPaths:
    """
    获取规范化路径
    
    Args:
        user_id: 用户 ID
        dataset_name: 数据集名称
        model_name: 模型名称 (lda, etm, ctm, dtm, stm, theta)
        timestamp: 时间戳 (默认自动生成)
        validate: 是否验证参数
    
    Returns:
        CanonicalPaths 对象，包含所有规范化路径
    
    Example:
        paths = get_canonical_paths("test_user", "policy_docs", "lda")
        paths.ensure_dirs()  # 创建所有目录
        
        # 使用路径
        bow_matrix = np.load(paths.bow_matrix_path)
        plt.savefig(paths.visualization_dir / "wordcloud.png")
    """
    if validate:
        user_id = validate_user_id(user_id)
        dataset_name = validate_dataset_name(dataset_name)
        model_name = validate_model_name(model_name)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return CanonicalPaths(
        user_id=user_id,
        dataset_name=dataset_name,
        model_name=model_name,
        timestamp=timestamp
    )


def get_paths_from_env() -> CanonicalPaths:
    """
    从环境变量获取路径配置
    
    用于 DLC 容器内的脚本，直接读取注入的环境变量
    
    Required env vars:
        - USERNAME or THETA_USER_ID
        - DATASET_NAME or THETA_DATASET
        - MODEL_TYPE
        - RUN_ID or TIMESTAMP
    """
    user_id = os.getenv("USERNAME") or os.getenv("THETA_USER_ID")
    dataset_name = os.getenv("DATASET_NAME") or os.getenv("THETA_DATASET")
    model_name = os.getenv("MODEL_TYPE", "lda")
    timestamp = os.getenv("RUN_ID") or os.getenv("TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not user_id:
        raise PathValidationError("USERNAME or THETA_USER_ID environment variable is required")
    if not dataset_name:
        raise PathValidationError("DATASET_NAME or THETA_DATASET environment variable is required")
    
    return get_canonical_paths(user_id, dataset_name, model_name, timestamp)


def find_input_csv(paths: CanonicalPaths) -> Optional[Path]:
    """
    查找输入 CSV 文件
    
    搜索顺序:
        1. processed_data.csv (优先使用已清洗的数据)
        2. data.csv
        3. 任意 .csv 文件
    """
    # 优先使用已处理的数据
    if paths.input_processed_csv.exists():
        return paths.input_processed_csv
    
    # 其次使用原始数据
    if paths.input_data_csv.exists():
        return paths.input_data_csv
    
    # 最后搜索任意 CSV
    csv_files = list(paths.input_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    
    return None


# 便捷函数：获取预训练模型路径
def get_model_paths() -> dict:
    """获取预训练模型路径"""
    return {
        "sbert": Path(MODELS_MOUNT) / "sbert",
        "qwen_0.6b": Path(MODELS_MOUNT) / "qwen-0.6b",
        "qwen_4b": Path(MODELS_MOUNT) / "qwen-4b",
        "qwen_8b": Path(MODELS_MOUNT) / "qwen-8b",
    }


# 便捷函数：获取代码路径
def get_code_paths() -> dict:
    """获取代码路径"""
    return {
        "theta_code": Path(CODE_MOUNT) / "src" / "models",
        "etm_code": Path(CODE_MOUNT) / "ETM",
        "dataclean": Path(CODE_MOUNT) / "src" / "models" / "dataclean",
    }
