#!/usr/bin/env python3
"""
THETA Training Engine v2 - DLC 容器入口脚本
与旧命令完全一致的流程

功能流程:
0. 强制依赖自愈 (Self-Healing)
1. 安装依赖
2. 读取环境变量获取用户配置
3. 创建统一目录结构
4. 格式转换: 将用户上传的文件转换为统一 CSV (dataclean)
5. 数据预处理: 生成 BOW + SBERT + Word2Vec + Qwen 嵌入 (prepare_data_unified)
6. 模型训练: 调用 run_pipeline.py
7. 生成可视化: WordCloud + Topic Distribution
8. 实时写入 training_log.json
9. 回调通知: 通知服务器训练完成

OSS 挂载结构 (新版):
    /mnt/code/      <- oss://bucket/code/       (THETA 代码)
    /mnt/models/    <- oss://bucket/models/     (qwen-0.6b/, sbert/)
    /mnt/raw_data/  <- oss://bucket/raw_data/   (用户上传的原始数据)
    /mnt/results/   <- oss://bucket/results/    (训练结果输出)

目录结构 (用户隔离):
    /mnt/raw_data/{username}/{dataset}/              # 原始上传数据
    /mnt/results/{username}/{dataset}/bow/           # BOW 和词汇表
    /mnt/results/{username}/{dataset}/embeddings/    # 所有嵌入
    /mnt/results/{username}/{dataset}/model/         # 模型权重
    /mnt/results/{username}/{dataset}/evaluation/    # 评估结果
    /mnt/results/{username}/{dataset}/visualization/ # 可视化
"""

# =============================================================================
# [PHASE 1] 强制路径注入 (PATH INJECTION FIRST)
# 必须在任何 import src... 之前执行，确保无论容器在哪启动，src 都能被精准定位
# =============================================================================
import sys
import os
import subprocess
import importlib.util

# [CRITICAL] 设置 DLC 环境标志，确保 config.py 识别为 DLC 环境
os.environ["DLC_ENV"] = "1"
print("[DLC_ENV] Set DLC_ENV=1 for DLC environment detection")

# 找到根目录（脚本所在目录）
_SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPT_ROOT)
    print(f"[SYS.PATH] Injected script root: {_SCRIPT_ROOT}")

# =============================================================================
# [OSS 模型路径] 统一配置 (固定 OSS 挂载位置)
# =============================================================================
# Qwen 嵌入模型路径 (按 model_size 区分)
QWEN_MODEL_PATHS = {
    "0.6B": "/mnt/models/qwen-0.6b/Qwen/Qwen3-Embedding-0__6B",
    "4B": "/mnt/models/qwen-4b/Qwen/Qwen3-Embedding-4B",
    "8B": "/mnt/models/qwen-8b/Qwen/Qwen3-Embedding-8B",
}
# SBERT 嵌入模型路径 (CTM/DTM 使用)
SBERT_MODEL_PATH = "/mnt/models/sbert/sentence-transformers/all-MiniLM-L6-v2"

# 注入 DLC 容器标准路径 (不包含 ETM，OSS 上不存在)
_DLC_PATHS = [
    "/mnt/code/src/models",    # THETA 模型代码
    "/mnt/code",               # 代码根目录
]
for _path in _DLC_PATHS:
    if _path not in sys.path and os.path.isdir(_path):
        sys.path.insert(0, _path)
        print(f"[SYS.PATH] Injected DLC path: {_path}")

# =============================================================================
# [PHASE 2] 依赖映射表 (THE DEPENDENCY MAP)
# 映射 导入名 -> pip安装名，支持包名和导入名不同的情况
# =============================================================================
DEPENDENCY_MAP = {
    # === 文档处理 ===
    "docx": "python-docx",           # Word 文档 (.docx)
    "pdf2docx": "pdf2docx",          # PDF 转 Word
    "fitz": "pymupdf",               # PDF 处理 (PyMuPDF)
    "pdfminer": "pdfminer.six",      # PDF 文本提取
    "PyPDF2": "PyPDF2",              # PDF 处理
    "openpyxl": "openpyxl",          # Excel 处理
    
    # === NLP 处理 ===
    "jieba": "jieba",                # 中文分词
    "gensim": "gensim",              # Word2Vec / Topic Models
    
    # === 数据处理 ===
    "pandas": "pandas",              # 数据处理
    "numpy": "numpy",                # 数值计算
    "scipy": "scipy",                # 科学计算
    "sklearn": "scikit-learn",       # 机器学习
    
    # === 可视化 ===
    "wordcloud": "wordcloud",        # 词云
    "matplotlib": "matplotlib",      # 绑图
    "seaborn": "seaborn",            # 统计可视化
    
    # === 工具库 ===
    "tqdm": "tqdm",                  # 进度条
    "click": "click",                # CLI 框架
    "requests": "requests",          # HTTP 请求
}

# 全局变量: 记录安装失败的关键依赖 (用于 Fallback 拦截)
_FAILED_CRITICAL_DEPS = []

# =============================================================================
# [PHASE 3] 哨兵函数 (THE SENTINEL)
# 使用 importlib.util.find_spec() 高效检测模块，一次性批量安装缺失包
# =============================================================================
def bootstrap_environment():
    """
    环境引导函数 (The Sentinel)
    
    功能:
    1. 使用 importlib.util.find_spec() 高效检测模块是否存在
    2. 收集所有缺失包，一次性执行 pip install (比逐个安装快得多)
    3. 记录安装失败的包，用于后续 Fallback 拦截
    
    Returns:
        bool: True if all dependencies are satisfied, False otherwise
    """
    global _FAILED_CRITICAL_DEPS
    _FAILED_CRITICAL_DEPS = []
    
    print("=" * 60)
    print("[BOOTSTRAP] Environment Initialization (The Sentinel)")
    print("=" * 60)
    
    # 使用 find_spec 高效检测模块
    missing_packages = []
    installed_modules = []
    
    for import_name, pip_name in DEPENDENCY_MAP.items():
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            missing_packages.append(pip_name)
            print(f"  [MISSING] {import_name} -> {pip_name}")
        else:
            installed_modules.append(import_name)
    
    print(f"  [SUMMARY] Installed: {len(installed_modules)}, Missing: {len(missing_packages)}")
    
    if not missing_packages:
        print("[BOOTSTRAP] All dependencies satisfied. Ready to go!")
        return True
    
    # 一次性批量安装所有缺失包
    print(f"\n[BOOTSTRAP] Installing {len(missing_packages)} packages in one batch...")
    print(f"  Packages: {', '.join(missing_packages)}")
    
    pip_cmd = [
        sys.executable, "-m", "pip", "install",
        "-q",                                    # 静默模式
        "--default-timeout=120",                 # 超时设置
        "-i", "https://mirrors.aliyun.com/pypi/simple/",  # 阿里云镜像
        "--trusted-host", "mirrors.aliyun.com",
    ] + missing_packages
    
    try:
        subprocess.check_call(pip_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"[BOOTSTRAP] Successfully installed: {', '.join(missing_packages)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[BOOTSTRAP] WARNING: Batch install failed, trying requirements.txt...")
        
        # 回退: 尝试从 requirements.txt 安装
        if _install_from_requirements():
            return True
        
        # 记录失败的包
        _FAILED_CRITICAL_DEPS = missing_packages.copy()
        print(f"[BOOTSTRAP] CRITICAL: Failed to install: {', '.join(_FAILED_CRITICAL_DEPS)}")
        return False


def _install_from_requirements():
    """
    从 requirements.txt 安装依赖 (回退方案)
    
    Returns:
        bool: 安装是否成功
    """
    requirements_paths = [
        "/mnt/code/requirements.txt",
        "/mnt/code/src/models/requirements.txt",
        os.path.join(_SCRIPT_ROOT, "requirements.txt"),
    ]
    
    for req_path in requirements_paths:
        if os.path.exists(req_path):
            print(f"[BOOTSTRAP] Found requirements.txt: {req_path}")
            cmd = [
                sys.executable, "-m", "pip", "install",
                "-q", "--default-timeout=120",
                "-i", "https://mirrors.aliyun.com/pypi/simple/",
                "--trusted-host", "mirrors.aliyun.com",
                "-r", req_path
            ]
            try:
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(f"[BOOTSTRAP] Installed from {req_path}")
                return True
            except subprocess.CalledProcessError:
                print(f"[BOOTSTRAP] WARNING: Failed to install from {req_path}")
    
    return False


# =============================================================================
# [PHASE 4] 立即执行环境引导 (在任何 src 模块导入之前)
# =============================================================================
_BOOTSTRAP_SUCCESS = bootstrap_environment()
print("=" * 60)

# =============================================================================
# [PHASE 4.5] GPU 算力验证 (COMPUTE VALIDATION)
# 强制要求: ecs.gn8v-2x.12xlarge (2 * GU8T, 96GB, 48 vCPU, 256 GiB)
# 如果检测到非预期 GPU，直接报错终止任务
# =============================================================================
def validate_gpu_compute():
    """
    验证 GPU 算力是否符合预期
    
    预期配置: ecs.gn8v-2x.12xlarge
    - 2 * GU8T (NVIDIA A100 80GB 或等效)
    - cuda capability >= 8.0
    
    如果检测到 V100 (cuda capability 7.0) 或其他不兼容 GPU，直接报错
    """
    print("=" * 60)
    print("[GPU VALIDATION] 算力验证")
    print("=" * 60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("  [WARN] CUDA not available, running on CPU")
            return True
        
        gpu_count = torch.cuda.device_count()
        print(f"  GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            
            print(f"  GPU{i}: {gpu_name}")
            print(f"    CUDA Capability: {capability[0]}.{capability[1]}")
            print(f"    Memory: {total_memory:.1f} GB")
            
            # 检查 CUDA capability
            # V100 = 7.0, T4 = 7.5, A10 = 8.6, A100 = 8.0, H100 = 9.0
            # PyTorch 2.1 支持 CUDA capability >= 7.0 (V100)
            cuda_version = capability[0] + capability[1] / 10
            if cuda_version < 7.0:
                error_msg = (
                    f"\n[FATAL] GPU 算力不符合要求!\n"
                    f"  检测到: {gpu_name} (CUDA capability {capability[0]}.{capability[1]})\n"
                    f"  要求: CUDA capability >= 7.0\n"
                    f"  预期实例: ecs.gn6e-c12g1.3xlarge (V100)\n"
                    f"\n"
                    f"  请检查 DLC 任务配置中的 ecs_spec 是否正确\n"
                )
                print(error_msg)
                raise RuntimeError(error_msg)
        
        print(f"  [OK] GPU 算力验证通过")
        return True
        
    except ImportError:
        print("  [WARN] PyTorch not installed, skipping GPU validation")
        return True
    except Exception as e:
        if "算力不符合要求" in str(e):
            raise
        print(f"  [WARN] GPU validation error: {e}")
        return True

# 执行 GPU 验证
validate_gpu_compute()
print("=" * 60)

# =============================================================================
# [PHASE 5] 延迟加载标准库 (在 bootstrap 之后)
# =============================================================================
import json
import shutil
import requests
from datetime import datetime
from pathlib import Path

# 代码目录
THETA_CODE_DIR = "/mnt/code/src/models"
ETM_CODE_DIR = "/mnt/code/ETM"

# 最小文件数量要求
MIN_DOCUMENT_COUNT = 5

# =============================================================================
# [ADMISSION CONTROL] 数据准入校验
# =============================================================================

class AdmissionError(Exception):
    """数据准入校验失败异常"""
    pass

class SchemaError(Exception):
    """CSV 格式/列名校验失败异常"""
    pass


def validate_data_admission(input_dir: str, model_type: str = "lda") -> dict:
    """
    硬性准入拦截 (Admission Control)
    
    校验规则:
    1. 有效文档数量 >= 5 (保证研究多样性与统计显著性)
    2. DTM/STM 模型需要 CSV 格式并包含特定列
    
    Args:
        input_dir: 输入数据目录 (通常为 /mnt/raw_data/{user}/{dataset}/)
        model_type: 模型类型 (lda, dtm, stm, etm, ctm, etc.)
    
    Returns:
        dict: 校验结果 {"valid": bool, "file_count": int, "files": list, "csv_file": str}
    
    Raises:
        AdmissionError: 文件数量不足
        SchemaError: CSV 格式/列名不符合要求
    """
    print("=" * 60)
    print("[ADMISSION CONTROL] 数据准入校验")
    print("=" * 60)
    
    input_path = Path(input_dir)
    
    # 支持的文件格式
    valid_extensions = {'.docx', '.pdf', '.txt', '.csv', '.xlsx'}
    
    # 统计有效文件
    valid_files = []
    csv_files = []
    
    for ext in valid_extensions:
        files = list(input_path.glob(f"*{ext}")) + list(input_path.glob(f"*{ext.upper()}"))
        valid_files.extend(files)
        if ext == '.csv':
            csv_files.extend(files)
    
    # 去重
    valid_files = list(set(valid_files))
    csv_files = list(set(csv_files))
    
    file_count = len(valid_files)
    
    print(f"  输入目录: {input_dir}")
    print(f"  有效文件数量: {file_count}")
    print(f"  CSV 文件数量: {len(csv_files)}")
    
    # === 规则 1: 最小文件数量校验 ===
    if file_count < MIN_DOCUMENT_COUNT:
        error_msg = (
            f"[CRITICAL] 初始数据源不足（当前为 {file_count} 个文件）。\n"
            f"THETA 系统要求至少上传 {MIN_DOCUMENT_COUNT} 个独立文档以保证研究的多样性与统计显著性。\n"
            f"支持的格式: .docx, .pdf, .txt, .csv, .xlsx"
        )
        print(f"\n{error_msg}")
        raise AdmissionError(error_msg)
    
    print(f"  [OK] 文件数量校验通过 ({file_count} >= {MIN_DOCUMENT_COUNT})")
    
    # === 规则 2: DTM/STM 模型特定校验 ===
    model_type_lower = model_type.lower()
    
    if model_type_lower in ('dtm', 'stm'):
        print(f"\n  [INFO] 检测到 {model_type.upper()} 模型，执行结构化数据校验...")
        
        # 必须有 CSV 文件
        if not csv_files:
            error_msg = (
                f"[ERROR] {model_type.upper()} 需要结构化数据，请提供 CSV 格式文件。\n"
                f"当前目录仅包含非 CSV 文件，无法进行 {model_type.upper()} 建模。\n"
                f"请将数据整理为 CSV 格式后重新上传。"
            )
            print(f"\n{error_msg}")
            raise SchemaError(error_msg)
        
        # 使用最大的 CSV 文件进行校验
        primary_csv = max(csv_files, key=lambda f: f.stat().st_size)
        print(f"  主 CSV 文件: {primary_csv.name}")
        
        # 读取 CSV 表头
        import pandas as pd
        try:
            df = pd.read_csv(primary_csv, nrows=5, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(primary_csv, nrows=5, encoding='gbk')
        
        columns = [col.lower().strip() for col in df.columns]
        print(f"  CSV 列名: {list(df.columns)}")
        
        # DTM 校验: 必须包含时间列
        if model_type_lower == 'dtm':
            time_keywords = {'time', 'year', 'date', 'timestamp', 'period', 'month', 'day', '时间', '年份', '日期'}
            has_time_col = any(any(kw in col for kw in time_keywords) for col in columns)
            
            if not has_time_col:
                error_msg = (
                    f"[ERROR] DTM (动态主题模型) 需要时间序列数据。\n"
                    f"当前 CSV 列名: {list(df.columns)}\n"
                    f"请确保 CSV 包含以下任一时间列: time, year, date, timestamp, period\n"
                    f"示例表头: id, text, year"
                )
                print(f"\n{error_msg}")
                raise SchemaError(error_msg)
            
            print(f"  [OK] DTM 时间列校验通过")
        
        # STM 校验: 必须包含协变量列 (除 id/text 外至少一列)
        elif model_type_lower == 'stm':
            # 排除常见的文本/ID 列
            text_id_keywords = {'id', 'text', 'content', 'document', 'doc', 'body', '文本', '内容', '正文'}
            covariate_cols = [col for col in columns if not any(kw in col for kw in text_id_keywords)]
            
            if len(covariate_cols) < 1:
                error_msg = (
                    f"[ERROR] STM (结构化主题模型) 需要协变量/元数据列。\n"
                    f"当前 CSV 列名: {list(df.columns)}\n"
                    f"请确保 CSV 除 id/text 外至少包含一列协变量 (如: author, category, source, year)\n"
                    f"示例表头: id, text, author, category"
                )
                print(f"\n{error_msg}")
                raise SchemaError(error_msg)
            
            print(f"  [OK] STM 协变量列校验通过 (协变量: {covariate_cols})")
        
        return {
            "valid": True,
            "file_count": file_count,
            "files": [str(f) for f in valid_files],
            "csv_file": str(primary_csv),
            "model_type": model_type
        }
    
    # 非 DTM/STM 模型，只需文件数量校验
    print(f"\n[ADMISSION CONTROL] 校验通过 ✓")
    return {
        "valid": True,
        "file_count": file_count,
        "files": [str(f) for f in valid_files],
        "csv_file": str(csv_files[0]) if csv_files else None,
        "model_type": model_type
    }


def install_dependencies():
    """
    安装 Python 依赖 (优化版)
    优先从 /mnt/code/requirements.txt 安装完整依赖
    PyTorch 官方镜像已包含: torch, numpy, scipy, scikit-learn, pandas, tqdm, matplotlib
    """
    print("=== Installing dependencies (from requirements.txt) ===")
    
    # DLC 容器内 requirements.txt 绝对路径
    requirements_path = "/mnt/code/requirements.txt"
    
    pip_opts = [
        "-q",  # 静默模式加速
        "--default-timeout=120",
        "-i", "https://mirrors.aliyun.com/pypi/simple/",
        "--trusted-host", "mirrors.aliyun.com"
    ]
    
    # 优先从 requirements.txt 安装
    if os.path.exists(requirements_path):
        print(f"[INFO] Found requirements.txt at: {requirements_path}")
        cmd = [sys.executable, "-m", "pip", "install"] + pip_opts + ["-r", requirements_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[WARN] Some dependencies failed: {result.stderr[:300]}")
        else:
            print("[OK] Dependencies installed from requirements.txt")
    else:
        # 回退: 只安装关键缺失包
        print(f"[WARN] requirements.txt not found at {requirements_path}, installing minimal deps...")
        missing_deps = ["jieba", "gensim", "wordcloud", "python-docx", "PyPDF2", "openpyxl"]
        cmd = [sys.executable, "-m", "pip", "install"] + pip_opts + missing_deps
        print(f"Installing: {', '.join(missing_deps)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[WARN] Some dependencies failed: {result.stderr[:200]}")
        else:
            print("[OK] Minimal dependencies installed")


def get_config():
    """
    获取 DLC 注入的环境变量
    
    新版路径架构:
        输入区: /mnt/raw_data/{user_id}/{dataset_name}/
        输出区: /mnt/results/{user_id}/{dataset_name}/{model_name}/{timestamp}/
            - workspace/: BOW, embeddings
            - model/: 模型权重
            - evaluation/: 评估指标
            - visualization/: 可视化图表
    """
    username = os.getenv("USERNAME") or os.getenv("THETA_USER_ID", "test_user")
    dataset_name = os.getenv("DATASET_NAME") or os.getenv("THETA_DATASET", "test_dataset")
    model_type = os.getenv("MODEL_TYPE", "lda")
    run_id = os.getenv("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # 新版规范化路径 (优先使用 DLC 注入的环境变量)
    input_dir = os.getenv("INPUT_DIR", f"/mnt/raw_data/{username}/{dataset_name}")
    output_dir = os.getenv("OUTPUT_DIR", f"/mnt/results/{username}/{dataset_name}/{model_type}/{run_id}")
    workspace_dir = os.getenv("WORKSPACE_DIR", f"{output_dir}/workspace")
    model_dir = os.getenv("MODEL_DIR", f"{output_dir}/model")
    evaluation_dir = os.getenv("EVALUATION_DIR", f"{output_dir}/evaluation")
    visualization_dir = os.getenv("VISUALIZATION_DIR", f"{output_dir}/visualization")
    
    config = {
        "username": username,
        "dataset_name": dataset_name,
        "run_id": run_id,
        "job_id": os.getenv("JOB_ID") or os.getenv("THETA_JOB_ID"),
        "model_type": model_type,
        "model_size": os.getenv("MODEL_SIZE", "0.6B"),
        "mode": os.getenv("MODE") or os.getenv("THETA_MODE", "zero_shot"),
        "num_topics": int(os.getenv("NUM_TOPICS", "10")),
        "epochs": int(os.getenv("EPOCHS", "50")),
        "language": os.getenv("LANGUAGE", "chinese"),
        "vocab_size": int(os.getenv("VOCAB_SIZE", "5000")),
        "api_base_url": os.getenv("API_BASE_URL"),
        "secret_key": os.getenv("SECRET_KEY", ""),
        # === 新版规范化路径 ===
        "input_dir": input_dir,                    # 输入区根目录
        "output_dir": output_dir,                  # 输出区根目录
        "workspace_dir": workspace_dir,            # 中间产物 (BOW, embeddings)
        "model_dir": model_dir,                    # 模型权重
        "evaluation_dir": evaluation_dir,          # 评估指标
        "visualization_dir": visualization_dir,    # 可视化图表
        # === 兼容旧版路径别名 ===
        "user_data_dir": input_dir,
        "user_result_base": output_dir,
        "user_bow_dir": workspace_dir,
        "user_embedding_dir": workspace_dir,
        "user_model_dir": model_dir,
        "user_eval_dir": evaluation_dir,
        "user_vis_dir": visualization_dir,
        # === 预训练模型路径 (使用顶部统一常量) ===
        "sbert_path": SBERT_MODEL_PATH,
        "qwen_path": QWEN_MODEL_PATHS.get(os.getenv("MODEL_SIZE", "0.6B"), QWEN_MODEL_PATHS["0.6B"]),
        "qwen_model_paths": QWEN_MODEL_PATHS,  # 完整路径映射
    }
    
    return config


def setup_directories(config: dict):
    """
    创建统一目录结构
    
    新版路径架构:
        输入区 (只读): /mnt/raw_data/{user_id}/{dataset_name}/
        输出区 (写入): /mnt/results/{user_id}/{dataset_name}/{model_name}/{timestamp}/
            - workspace/: BOW, embeddings
            - model/: 模型权重
            - evaluation/: 评估指标
            - visualization/: 可视化图表
    """
    print("=== Creating unified directory structure ===")
    
    # 只创建输出区目录 (输入区是只读的)
    dirs = [
        config["workspace_dir"],
        config["model_dir"],
        config["evaluation_dir"],
        config["visualization_dir"],
    ]
    
    for path in dirs:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    print(f"  Input (read):  {config['input_dir']}")
    print(f"  Output:        {config['output_dir']}")
    print(f"    workspace/:  {config['workspace_dir']}")
    print(f"    model/:      {config['model_dir']}")
    print(f"    evaluation/: {config['evaluation_dir']}")
    print(f"    visualization/: {config['visualization_dir']}")


def check_and_prepare_data(config: dict) -> str:
    """
    Step 0: 确定主数据文件 (与旧命令一致)
    优先使用已有的 CSV 文件，只有在没有 CSV 时才转换非 CSV 文件
    """
    print("=== Step 0: Determining primary data file ===")
    
    data_dir = Path(config["user_data_dir"])
    output_csv = data_dir / "data.csv"
    
    # 查找已有的 CSV 文件
    csv_files = list(data_dir.glob("*.csv"))
    non_csv_files = list(data_dir.glob("*.docx")) + list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.txt"))
    
    print(f"Found {len(csv_files)} CSV files, {len(non_csv_files)} non-CSV files")
    
    if csv_files:
        # 使用最大的 CSV 文件作为主数据文件
        primary_csv = max(csv_files, key=lambda f: f.stat().st_size)
        print(f"Using existing CSV as primary data: {primary_csv}")
        
        if primary_csv != output_csv:
            shutil.copy(primary_csv, output_csv)
            print(f"Copied to {output_csv}")
    elif non_csv_files:
        print("No CSV files found, converting non-CSV files...")
        
        dataclean_main = Path(THETA_CODE_DIR) / "dataclean" / "main.py"
        if dataclean_main.exists():
            cmd = [
                "python", "-m", "dataclean.main", "convert",
                str(data_dir), str(output_csv),
                "--language", "chinese" if config["language"] == "chinese" else "english",
                "--recursive"
            ]
            subprocess.run(cmd, cwd=THETA_CODE_DIR)
        else:
            # 回退方案
            _fallback_convert(non_csv_files, output_csv)
        
        print(f"Conversion completed! Output: {output_csv}")
    else:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    # 显示主数据文件信息
    if output_csv.exists():
        with open(output_csv, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"Primary data file: {len(lines)} lines")
        for line in lines[:3]:
            print(f"  {line.strip()[:100]}...")
    
    return str(output_csv)


def _fallback_convert(files: list, output_csv: Path):
    """
    回退转换方案
    
    CRITICAL: 如果关键依赖安装失败，禁止处理 PDF/DOCX 文件
    直接抛出明确错误，防止乱码数据进入模型
    """
    import csv
    
    # 检查是否有关键依赖安装失败
    if _FAILED_CRITICAL_DEPS:
        # 检查文件类型，如果包含 PDF/DOCX 则拒绝处理
        binary_extensions = {'.pdf', '.docx', '.doc', '.xlsx', '.xls'}
        binary_files = [f for f in files if Path(f).suffix.lower() in binary_extensions]
        
        if binary_files:
            error_msg = (
                f"[CRITICAL] 无法处理二进制文件，因为关键依赖安装失败！\n"
                f"  缺失的库: {', '.join(_FAILED_CRITICAL_DEPS)}\n"
                f"  受影响的文件 ({len(binary_files)} 个):\n"
            )
            for bf in binary_files[:5]:
                error_msg += f"    - {Path(bf).name}\n"
            if len(binary_files) > 5:
                error_msg += f"    ... 还有 {len(binary_files) - 5} 个文件\n"
            error_msg += (
                f"\n请确保以下库已正确安装:\n"
                f"  - python-docx (用于 .docx 文件)\n"
                f"  - pdfminer.six (用于 .pdf 文件)\n"
                f"  - PyPDF2 (用于 .pdf 文件)\n"
                f"  - openpyxl (用于 .xlsx 文件)\n"
                f"\n禁止使用 Fallback 处理二进制文件，防止乱码数据进入模型！"
            )
            print(error_msg)
            raise RuntimeError(error_msg)
    
    # 只处理纯文本文件
    text_extensions = {'.txt', '.csv', '.json', '.xml', '.html', '.md'}
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "source_file"])
        
        doc_id = 1
        for file_path in files:
            ext = Path(file_path).suffix.lower()
            
            # 检查是否为二进制文件
            if ext in {'.pdf', '.docx', '.doc', '.xlsx', '.xls'}:
                print(f"  [SKIP] 跳过二进制文件 (无法在 Fallback 模式下处理): {Path(file_path).name}")
                continue
            
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as rf:
                    content = rf.read()
                writer.writerow([doc_id, content.strip(), Path(file_path).name])
                doc_id += 1
            except Exception as e:
                print(f"  [WARN] Failed to read {file_path}: {e}")
    
    if doc_id == 1:
        raise RuntimeError(
            "[CRITICAL] Fallback 转换失败：没有成功处理任何文件！\n"
            "请检查输入文件格式和依赖安装状态。"
        )


def prepare_data_unified(config: dict) -> bool:
    """
    Step 1 & 2: 统一数据预处理 (与旧命令一致)
    生成 BOW + SBERT + Word2Vec + Qwen 嵌入
    
    Returns:
        bool: True if successful, False if failed
    """
    print("=== Step 1 & 2: Unified Data Preparation ===")
    
    # 设置 SBERT 模型路径 (新版目录结构)
    # 实际 OSS 结构: /mnt/models/sbert/sentence-transformers/all-MiniLM-L6-v2/
    sbert_candidates = [
        "/mnt/models/sbert/sentence-transformers/all-MiniLM-L6-v2",  # OSS 实际路径
        "/mnt/models/sbert/all-MiniLM-L6-v2",
        "/mnt/models/sbert",
        config.get("sbert_path", "/mnt/models/sbert"),
    ]
    sbert_path = None
    for candidate in sbert_candidates:
        if Path(candidate).exists() and (Path(candidate) / "config.json").exists():
            sbert_path = candidate
            break
    
    if sbert_path:
        os.environ["SBERT_MODEL_PATH"] = sbert_path
        print(f"  SBERT model: {sbert_path}")
    else:
        print(f"  [WARN] SBERT model not found in any candidate paths:")
        for c in sbert_candidates:
            print(f"    {'✓' if Path(c).exists() else '✗'} {c}")
    
    # 检查 Qwen 模型 (新版目录结构)
    qwen_path = config.get("qwen_path", "/mnt/models/qwen-0.6b")
    generate_qwen = Path(qwen_path).exists()
    print(f"  Generate Qwen embeddings: {generate_qwen}")
    
    # 调用 prepare_data_unified
    sys.path.insert(0, THETA_CODE_DIR)
    sys.path.insert(0, ETM_CODE_DIR)
    
    # [导入路径加固] 显式检查 prepare_data 模块是否可导入
    print(f"[DEBUG] Checking prepare_data module availability...")
    print(f"  THETA_CODE_DIR: {THETA_CODE_DIR}")
    print(f"  sys.path[:5]: {sys.path[:5]}")
    
    prepare_data_path = Path(THETA_CODE_DIR) / "prepare_data.py"
    if not prepare_data_path.exists():
        print(f"  [WARN] prepare_data.py not found at: {prepare_data_path}")
        # 尝试其他路径
        alt_paths = [
            "/mnt/code/src/models/prepare_data.py",
            "/mnt/code/ETM/prepare_data.py",
        ]
        for alt in alt_paths:
            if Path(alt).exists():
                sys.path.insert(0, str(Path(alt).parent))
                print(f"  [OK] Found prepare_data.py at: {alt}")
                break
    else:
        print(f"  [OK] prepare_data.py found at: {prepare_data_path}")
    
    success = False
    try:
        from prepare_data import prepare_data_unified as _prepare_data_unified
        
        _prepare_data_unified(
            user_id=config["username"],
            dataset=config["dataset_name"],
            mount_point="/mnt",
            vocab_size=config["vocab_size"],
            language="zh" if config["language"] == "chinese" else "en",
            generate_qwen=generate_qwen,
            model_size=config["model_size"]
        )
        print("Unified data preparation completed!")
        success = True
    except ImportError:
        print("[WARN] prepare_data_unified not found, using fallback")
        success = _prepare_data_fallback(config)
    except Exception as e:
        print(f"[ERROR] Data preparation failed: {e}")
        success = False
    
    # 验证输出
    print("=== Verifying BOW files ===")
    bow_dir = Path(config["user_bow_dir"])
    bow_files = list(bow_dir.glob("*"))
    for f in bow_files:
        print(f"  {f.name}")
    
    # 也检查 user_result_base 目录
    result_base = Path(config["user_result_base"])
    result_bow_files = list(result_base.glob("bow_matrix.*"))
    for f in result_bow_files:
        print(f"  {f.name} (in result_base)")
    
    print("=== Verifying embedding files ===")
    emb_dir = Path(config["user_embedding_dir"])
    for f in emb_dir.glob("*"):
        print(f"  {f.name}")
    
    # 最终验证
    if not bow_files and not result_bow_files:
        print("[ERROR] No BOW files generated - data preparation failed!")
        return False
    
    return success


def _prepare_data_fallback(config: dict) -> bool:
    """
    数据预处理回退方案
    
    新版路径架构:
        输入区: INPUT_DIR = /mnt/raw_data/{user}/{dataset}/
        输出区: WORKSPACE_DIR = /mnt/results/{user}/{dataset}/{model}/{timestamp}/workspace/
    
    Returns:
        bool: True if successful, False if failed
    """
    prepare_script = Path(THETA_CODE_DIR) / "prepare_data.py"
    
    if not prepare_script.exists():
        print(f"  [ERROR] prepare_data.py not found: {prepare_script}")
        return False
    
    # 根据模型类型选择预处理模式
    # THETA 模型需要生成 Qwen embeddings
    model_type = config.get("model_type", "theta")
    print(f"  [DEBUG] model_type from config: {model_type}")
    if model_type == "theta":
        prepare_model = "theta"
        model_size = config.get("model_size", "0.6B")
        mode = config.get("mode", "zero_shot")
    else:
        prepare_model = "baseline"
        model_size = "0.6B"
        mode = "zero_shot"
    
    # 使用新版路径架构
    cmd = [
        "python", str(prepare_script),
        "--dataset", config["dataset_name"],
        "--model", prepare_model,
        "--model_size", model_size,
        "--mode", mode,
        "--vocab_size", str(config["vocab_size"]),
        "--language", "chinese" if config["language"] == "chinese" else "english",
        "--user_id", config["username"],
        "--output_dir", config["workspace_dir"],  # BOW 输出到 workspace/
    ]
    print(f"  [EXEC] {' '.join(cmd)}")
    
    # [CRITICAL] 传递 DLC 环境变量给子进程
    env = os.environ.copy()
    env["DLC_ENV"] = "1"
    env["INPUT_DIR"] = config["user_data_dir"]
    env["OUTPUT_DIR"] = config["output_dir"]
    print(f"  [ENV] DLC_ENV=1, INPUT_DIR={config['user_data_dir']}")
    
    result = subprocess.run(cmd, cwd=THETA_CODE_DIR, capture_output=True, text=True, env=env)
    print(result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout)
    
    if result.returncode != 0:
        print(f"  [ERROR] prepare_data failed: {result.stderr[-500:]}")
        return False
    
    # 检查是否包含 "Data file not found" 错误
    if "Data file not found" in result.stdout:
        print(f"  [ERROR] Data file not found - check path configuration")
        return False
    
    # 验证 BOW 文件是否生成
    bow_dir = Path(config["user_result_base"])
    bow_files = list(bow_dir.glob("bow_matrix.*")) + list(bow_dir.glob("*/bow_matrix.*"))
    if not bow_files:
        print(f"  [ERROR] BOW matrix not generated in {bow_dir}")
        return False
    
    print(f"  [OK] Data preparation completed")
    return True


def run_training(config: dict):
    """
    Step 3: 模型训练
    
    新版路径架构:
        --workspace_dir: workspace/ 目录 (读取 BOW)
        --output_dir: output/ 根目录 (写入模型、评估、可视化)
    """
    print(f"=== Step 3: Model Training ({config['model_type']}) ===")
    
    pipeline_script = Path(THETA_CODE_DIR) / "run_pipeline.py"
    
    cmd = [
        "python", str(pipeline_script),
        "--dataset", config["dataset_name"],
        "--models", config["model_type"],
        "--model_size", config["model_size"],
        "--mode", config["mode"],
        "--num_topics", str(config["num_topics"]),
        "--epochs", str(config["epochs"]),
        "--gpu", "0",
        "--user_id", config["username"],
        "--workspace_dir", config["workspace_dir"],      # 读取 BOW 从 workspace/
        "--output_dir", config["output_dir"],            # 输出到 output/ 根目录
        "--lang", "cn" if config["language"] == "chinese" else "en",
    ]
    
    print(f"  [EXEC] {' '.join(cmd)}")
    
    # [CRITICAL] 传递 DLC 环境变量给子进程
    env = os.environ.copy()
    env["DLC_ENV"] = "1"
    env["INPUT_DIR"] = config["user_data_dir"]
    env["OUTPUT_DIR"] = config["output_dir"]
    env["WORKSPACE_DIR"] = config["workspace_dir"]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=THETA_CODE_DIR, env=env)
    
    if result.returncode != 0:
        print(f"  [ERROR] Training failed: {result.stderr}")
        raise RuntimeError(f"Training failed: {result.stderr}")
    
    print(result.stdout)
    print("  [OK] Training completed")


def generate_visualizations(config: dict):
    """
    Step 4: 生成可视化 (与旧命令一致)
    WordCloud + Topic Distribution
    """
    print("=== Step 4: Generating Visualizations ===")
    
    result_dir = Path(config["user_result_base"])
    model_dir = Path(config["user_model_dir"])
    vis_dir = Path(config["user_vis_dir"])
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        
        # 查找 topic_words 文件
        topic_words_file = None
        for f in list(model_dir.glob("*topic_words*.json")) + list(result_dir.glob("**/topic_words*.json")):
            topic_words_file = f
            break
        
        if topic_words_file:
            with open(topic_words_file, 'r', encoding='utf-8') as f:
                topic_words = json.load(f)
            
            topics = topic_words if isinstance(topic_words, list) else list(topic_words.values())
            
            for topic_id, words in enumerate(topics):
                if isinstance(words, list):
                    word_freq = {w: 1.0/(i+1) for i, w in enumerate(words[:20])}
                else:
                    word_freq = words
                
                # 尝试使用中文字体
                font_path = None
                for fp in ['/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                           '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf']:
                    if Path(fp).exists():
                        font_path = fp
                        break
                
                wc = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    font_path=font_path
                ).generate_from_frequencies(word_freq)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.savefig(vis_dir / f'wordcloud_topic_{topic_id}.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"  Generated {len(topics)} wordcloud images")
        
        # 生成 Topic Distribution
        theta_file = None
        for f in list(model_dir.glob("*theta*.npy")) + list(result_dir.glob("**/theta*.npy")):
            theta_file = f
            break
        
        if theta_file:
            theta = np.load(theta_file)
            topic_dist = theta.mean(axis=0)
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(topic_dist)), topic_dist)
            plt.xlabel('Topic')
            plt.ylabel('Average Probability')
            plt.title('Topic Distribution')
            plt.savefig(vis_dir / 'topic_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  Generated topic_distribution.png")
        
        print(f"  Output directory: {vis_dir}")
        
    except Exception as e:
        print(f"  [WARN] Visualization failed: {e}")


def save_training_log(config: dict, status: str):
    """保存训练日志"""
    log_file = Path(config["user_result_base"]) / "training_log.json"
    
    log_data = {
        "run_id": config["run_id"],
        "job_id": config["job_id"],
        "username": config["username"],
        "dataset_name": config["dataset_name"],
        "model_type": config["model_type"],
        "status": status,
        "completed_at": datetime.now().isoformat(),
        "config": {
            "num_topics": config["num_topics"],
            "epochs": config["epochs"],
            "vocab_size": config["vocab_size"],
            "model_size": config["model_size"],
            "mode": config["mode"],
        }
    }
    
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Training log saved: {log_file}")


def notify_server(config: dict, status: str):
    """通知服务器训练完成"""
    api_base_url = config.get("api_base_url")
    job_id = config.get("job_id")
    
    if not api_base_url or not job_id:
        print("[INFO] No callback URL configured, skipping")
        return
    
    callback_url = f"{api_base_url}/api/train/callback"
    payload = {
        "job_id": int(job_id) if job_id else 0,
        "status": status,
        "secret_key": config.get("secret_key", "")
    }
    
    try:
        response = requests.post(callback_url, json=payload, timeout=30)
        if response.status_code == 200:
            print(f"[OK] Server callback succeeded: {status}")
        else:
            print(f"[WARN] Server callback failed: {response.status_code}")
    except Exception as e:
        print(f"[WARN] Server callback error: {e}")


def print_final_summary(config: dict):
    """打印最终目录结构"""
    print("=== Training completed ===")
    print(f"User: {config['username']}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Mode: {config['mode']}")
    print("")
    print("=== Directory structure ===")
    
    dirs = [
        ("Data", config["user_data_dir"]),
        ("BOW", config["user_bow_dir"]),
        ("Embeddings", config["user_embedding_dir"]),
        ("Model", config["user_model_dir"]),
        ("Evaluation", config["user_eval_dir"]),
        ("Visualization", config["user_vis_dir"]),
    ]
    
    for name, path in dirs:
        print(f"{name}: {path}")
        p = Path(path)
        if p.exists():
            for f in list(p.iterdir())[:5]:
                print(f"  {f.name}")
        else:
            print("  (empty)")


def main():
    print("=" * 60)
    print("THETA Training Engine v2 - DLC Container")
    print("=" * 60)
    
    # 0. 安装依赖
    install_dependencies()
    
    # 1. 获取配置
    config = get_config()
    
    print(f"User: {config['username']}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Model: {config['model_type']}")
    print(f"Mode: {config['mode']}")
    print(f"Topics: {config['num_topics']}")
    print(f"Epochs: {config['epochs']}")
    
    # 2. 创建目录结构
    setup_directories(config)
    
    training_status = "succeeded"
    
    try:
        # === [ADMISSION CONTROL] 数据准入校验 (在任何数据处理之前) ===
        try:
            admission_result = validate_data_admission(
                input_dir=config["user_data_dir"],
                model_type=config["model_type"]
            )
            config["admission_result"] = admission_result
        except (AdmissionError, SchemaError) as e:
            # 准入校验失败，立即中止并通知服务器
            print(f"\n[FATAL] 数据准入校验失败:")
            print(str(e))
            training_status = "failed"
            config["error_message"] = str(e)
            save_training_log(config, training_status)
            notify_server(config, training_status)
            sys.exit(1)
        
        # 3. 检查并准备数据
        check_and_prepare_data(config)
        
        # 4. 统一数据预处理
        data_prep_success = prepare_data_unified(config)
        if not data_prep_success:
            raise RuntimeError("Data preparation failed - BOW matrix not generated")
        
        # 5. 模型训练
        run_training(config)
        
        # 6. 生成可视化
        generate_visualizations(config)
        
    except Exception as e:
        print(f"[FATAL] Training failed: {e}")
        training_status = "failed"
        import traceback
        traceback.print_exc()
    
    # 7. 保存训练日志
    save_training_log(config, training_status)
    
    # 8. 回调通知
    notify_server(config, training_status)
    
    # 9. 打印最终摘要
    print_final_summary(config)
    
    sys.exit(0 if training_status == "succeeded" else 1)


if __name__ == "__main__":
    main()
