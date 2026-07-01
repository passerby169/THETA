#!/usr/bin/env python3
"""
THETA Training Engine - DLC 容器入口脚本

功能流程:
1. 安装依赖
2. 读取环境变量获取用户配置
3. 创建统一目录结构
4. 格式转换: 将用户上传的文件转换为统一 CSV (dataclean)
5. 数据预处理: 生成 BOW + SBERT + Word2Vec + Qwen 嵌入
6. 模型训练: 调用 THETA 或基线模型训练 (run_pipeline.py)
7. 生成可视化: WordCloud + Topic Distribution
8. 实时写入 training_log.json
9. 回调通知: 通知服务器训练完成

OSS 挂载结构 (与旧命令一致):
    /mnt/code/              <- oss://bucket/code/        (THETA 代码)
    /mnt/{username}/        <- oss://bucket/{username}/  (用户隔离目录)
    /mnt/embedding_models/  <- oss://bucket/embedding_models/ (SBERT/Qwen)
    /mnt/sbert/             <- oss://bucket/sbert/       (SBERT 备用路径)

目录结构 (一用户一目录隔离 - 与旧命令一致):
    /mnt/{username}/data/{dataset}/              # 原始上传数据
    /mnt/{username}/result/{dataset}/bow/        # BOW 和词汇表
    /mnt/{username}/result/{dataset}/embeddings/ # 所有嵌入
    /mnt/{username}/result/{dataset}/model/      # 模型权重
    /mnt/{username}/result/{dataset}/evaluation/ # 评估结果
    /mnt/{username}/result/{dataset}/visualization/ # 可视化
"""

import os
import sys
import json
import subprocess
import requests
from datetime import datetime
from pathlib import Path

# 添加 THETA 代码路径
THETA_CODE_DIR = "/mnt/code/src/models"
sys.path.insert(0, THETA_CODE_DIR)


def install_dependencies():
    """安装 Python 依赖"""
    print("[STEP 0] 安装依赖")
    
    pip_opts = [
        "--default-timeout=1000",
        "--retries=10",
        "-i", "https://mirrors.aliyun.com/pypi/simple/",
        "--trusted-host", "mirrors.aliyun.com"
    ]
    
    # 核心依赖
    core_deps = [
        "transformers", "torch", "numpy", "scipy", "scikit-learn",
        "tqdm", "jieba", "pandas", "sentence-transformers", "gensim",
        "matplotlib", "wordcloud"
    ]
    
    # dataclean 依赖
    dataclean_deps = [
        "python-docx", "PyPDF2", "pdfminer.six", "click"
    ]
    
    all_deps = core_deps + dataclean_deps
    
    cmd = ["pip", "install"] + pip_opts + all_deps
    print(f"  [EXEC] pip install {' '.join(all_deps[:5])}...")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] 部分依赖安装失败: {result.stderr[:500]}")
    else:
        print("  [OK] 依赖安装完成")


def get_env_vars():
    """获取 DLC 注入的环境变量"""
    required_vars = {
        "USERNAME": os.getenv("USERNAME"),
        "DATASET_NAME": os.getenv("DATASET_NAME"),
    }

    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

    username = required_vars["USERNAME"]
    dataset_name = required_vars["DATASET_NAME"]

    return {
        "user_id": os.getenv("USER_ID"),
        "username": username,
        "dataset_name": dataset_name,
        "run_id": os.getenv("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S")),
        "model_type": os.getenv("MODEL_TYPE", "lda"),
        "model_size": os.getenv("MODEL_SIZE", "0.6B"),
        "mode": os.getenv("MODE", "zero_shot"),
        "num_topics": int(os.getenv("NUM_TOPICS", "20")),
        "epochs": int(os.getenv("EPOCHS", "100")),
        "language": os.getenv("LANGUAGE", "chinese"),
        "vocab_size": int(os.getenv("VOCAB_SIZE", "5000")),
        "job_id": os.getenv("JOB_ID"),
        "api_base_url": os.getenv("API_BASE_URL"),
        "secret_key": os.getenv("SECRET_KEY", ""),
        # 与旧命令一致的目录结构
        "user_base": f"/mnt/{username}",
        "input_file_path": f"/mnt/raw_data/{username}/{dataset_name}/",
        "result_path": f"/mnt/results/{username}/{dataset_name}/",
        "user_data_dir": f"/mnt/{username}/data/{dataset_name}",
        "user_result_base": f"/mnt/{username}/result/{dataset_name}",
        "user_bow_dir": f"/mnt/{username}/result/{dataset_name}/bow",
        "user_embedding_dir": f"/mnt/{username}/result/{dataset_name}/embeddings",
        "user_model_dir": f"/mnt/{username}/result/{dataset_name}/model",
        "user_eval_dir": f"/mnt/{username}/result/{dataset_name}/evaluation",
        "user_vis_dir": f"/mnt/{username}/result/{dataset_name}/visualization",
    }


def setup_directories(config: dict):
    """创建统一目录结构 (与旧命令一致)"""
    print("[STEP 1] 创建统一目录结构")

    # 定义标准目录结构
    user_base = Path(config["user_base"])
    workspace_dir = user_base / f"workspace/{config['dataset_name']}"
    result_dir = user_base / f"result/{config['dataset_name']}"

    dirs = {
        "data": config["user_data_dir"],
        "bow": config["user_bow_dir"],
        "embeddings": config["user_embedding_dir"],
        "model": config["user_model_dir"],
        "evaluation": config["user_eval_dir"],
        "visualization": config["user_vis_dir"],
    }

    for name, path in dirs.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"  {name}: {path}")

    workspace_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    return {
        "workspace_dir": workspace_dir,
        "result_dir": result_dir,
    }


def convert_to_csv(input_file: str, output_dir: Path, language: str) -> Path:
    """
    将用户上传的文件转换为统一 CSV 格式
    支持: txt, pdf, docx, xlsx, csv, json 等
    """
    print(f"[STEP 1] 格式转换: {input_file}")
    
    input_path = Path(input_file)
    output_csv = output_dir / f"{input_path.stem}_cleaned.csv"
    
    # 如果已经是 CSV，检查是否需要清洗
    if input_path.suffix.lower() == ".csv":
        # 直接复制或清洗
        import shutil
        shutil.copy(input_file, output_csv)
        print(f"  [INFO] CSV 文件已复制到: {output_csv}")
        return output_csv
    
    # 使用 dataclean 模块转换
    dataclean_main = Path(THETA_CODE_DIR) / "dataclean" / "main.py"
    
    if dataclean_main.exists():
        cmd = [
            "python", str(dataclean_main),
            "convert",
            str(input_file),
            str(output_csv),
            "--language", language,
            "--clean"
        ]
        
        print(f"  [EXEC] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  [WARN] dataclean 转换失败: {result.stderr}")
            # 回退: 简单读取文本
            _fallback_convert(input_file, output_csv)
        else:
            print(f"  [INFO] 转换成功: {output_csv}")
    else:
        print(f"  [WARN] dataclean 模块不存在，使用回退方案")
        _fallback_convert(input_file, output_csv)
    
    return output_csv


def _fallback_convert(input_file: str, output_csv: Path):
    """回退转换方案: 简单读取文本内容"""
    import csv
    
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])
        for i, line in enumerate(content.split("\n")):
            if line.strip():
                writer.writerow([i + 1, line.strip()])
    
    print(f"  [INFO] 回退转换完成: {output_csv}")


def prepare_data(config: dict, dirs: dict, csv_file: Path):
    """
    数据预处理: 生成 BOW 矩阵和嵌入向量
    """
    print(f"[STEP 2] 数据预处理")
    
    prepare_script = Path(THETA_CODE_DIR) / "prepare_data.py"
    
    cmd = [
        "python", str(prepare_script),
        "--dataset", config["dataset_name"],
        "--model", "baseline" if config["model_type"] != "theta" else "theta",
        "--model_size", config["model_size"],
        "--mode", config["mode"],
        "--vocab_size", str(config["vocab_size"]),
        "--language", config["language"],
        "--user_id", config["username"],
        "--output_dir", str(dirs["workspace_dir"]),
    ]
    
    print(f"  [EXEC] {' '.join(cmd)}")
    
    # 设置环境变量指向数据目录
    env = os.environ.copy()
    env["DATA_DIR"] = str(Path(config["input_file_path"]).parent.parent)
    env["RESULT_DIR"] = str(dirs["result_dir"].parent.parent.parent.parent)
    env["WORKSPACE_DIR"] = str(dirs["workspace_dir"].parent.parent)
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"  [ERROR] 预处理失败: {result.stderr}")
        raise RuntimeError(f"Data preparation failed: {result.stderr}")
    
    print(f"  [INFO] 预处理完成")
    print(result.stdout)


class TrainingLogger:
    """训练日志记录器 - 实时写入 training_log.json"""
    
    def __init__(self, result_dir: Path, run_id: str, total_epochs: int):
        self.log_file = result_dir / "training_log.json"
        self.run_id = run_id
        self.total_epochs = total_epochs
        self.metrics = []
        self._init_log()
    
    def _init_log(self):
        """初始化日志文件"""
        log_data = {
            "run_id": self.run_id,
            "status": "running",
            "current_epoch": 0,
            "total_epochs": self.total_epochs,
            "started_at": datetime.now().isoformat(),
            "metrics": []
        }
        self._write_log(log_data)
    
    def _write_log(self, data: dict):
        """写入日志文件"""
        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def log_epoch(self, epoch: int, loss: float, accuracy: float = None, **kwargs):
        """记录一个 epoch 的指标"""
        entry = {
            "epoch": epoch,
            "loss": loss,
            "timestamp": datetime.now().isoformat()
        }
        if accuracy is not None:
            entry["accuracy"] = accuracy
        entry.update(kwargs)
        self.metrics.append(entry)
        
        log_data = {
            "run_id": self.run_id,
            "status": "running",
            "current_epoch": epoch,
            "total_epochs": self.total_epochs,
            "started_at": self.metrics[0]["timestamp"] if self.metrics else datetime.now().isoformat(),
            "metrics": self.metrics
        }
        self._write_log(log_data)
    
    def complete(self, status: str = "succeeded"):
        """标记训练完成"""
        log_data = {
            "run_id": self.run_id,
            "status": status,
            "current_epoch": self.total_epochs if status == "succeeded" else len(self.metrics),
            "total_epochs": self.total_epochs,
            "started_at": self.metrics[0]["timestamp"] if self.metrics else datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "metrics": self.metrics
        }
        self._write_log(log_data)


def run_training(config: dict, dirs: dict):
    """
    运行模型训练
    """
    print(f"[STEP 3] 模型训练: {config['model_type']}")
    
    pipeline_script = Path(THETA_CODE_DIR) / "run_pipeline.py"
    
    # 使用 run_id 作为 task_name
    task_name = config["run_id"]
    
    cmd = [
        "python", str(pipeline_script),
        "--dataset", config["dataset_name"],
        "--models", config["model_type"],
        "--user_id", config["username"],
        "--task_name", task_name,
        "--num_topics", str(config["num_topics"]),
        "--epochs", str(config["epochs"]),
        "--vocab_size", str(config["vocab_size"]),
        "--lang", "cn" if config["language"] == "chinese" else "en",
    ]
    
    # THETA 特定参数
    if config["model_type"] == "theta":
        cmd.extend([
            "--model_size", config["model_size"],
            "--mode", config["mode"],
        ])
    
    print(f"  [EXEC] {' '.join(cmd)}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["DATA_DIR"] = str(Path(config["input_file_path"]).parent.parent)
    env["RESULT_DIR"] = str(dirs["result_dir"].parent.parent.parent.parent)
    env["WORKSPACE_DIR"] = str(dirs["workspace_dir"].parent.parent)
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"  [ERROR] 训练失败: {result.stderr}")
        raise RuntimeError(f"Training failed: {result.stderr}")
    
    print(f"  [INFO] 训练完成")
    print(result.stdout)
    
    return task_name


def save_job_result(config: dict, dirs: dict, status: str):
    """保存任务结果摘要"""
    result_file = dirs["result_dir"] / "job_result.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "job_id": config["job_id"],
        "username": config["username"],
        "run_id": config["run_id"],
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
    
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] 结果已保存: {result_file}")


def notify_server(config: dict, status: str):
    """通知服务器训练完成"""
    api_base_url = config.get("api_base_url")
    job_id = config.get("job_id")
    secret_key = config.get("secret_key")
    
    if not api_base_url or not job_id:
        print("[WARN] API_BASE_URL 或 JOB_ID 未设置，跳过回调")
        return False
    
    callback_url = f"{api_base_url}/api/train/callback"
    payload = {
        "job_id": int(job_id),
        "status": status,
        "secret_key": secret_key
    }
    
    try:
        response = requests.post(callback_url, json=payload, timeout=30)
        if response.status_code == 200:
            print(f"[INFO] 服务器回调成功: {status}")
            return True
        else:
            print(f"[WARN] 服务器回调失败: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] 服务器回调异常: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("THETA Training Engine - DLC Container")
    print("=" * 60)
    
    # 1. 获取配置
    try:
        config = get_env_vars()
    except ValueError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
    
    print(f"[CONFIG] USERNAME: {config['username']}")
    print(f"[CONFIG] DATASET: {config['dataset_name']}")
    print(f"[CONFIG] RUN_ID: {config['run_id']}")
    print(f"[CONFIG] MODEL: {config['model_type']}")
    print(f"[CONFIG] INPUT: {config['input_file_path']}")
    print(f"[CONFIG] RESULT_PATH: {config['result_path']}")
    
    # 2. 创建目录结构
    dirs = setup_directories(config)
    
    # 3. 初始化训练日志
    logger = TrainingLogger(dirs["result_dir"], config["run_id"], config["epochs"])
    
    training_status = "succeeded"
    
    try:
        # 4. 格式转换
        csv_file = convert_to_csv(
            config["input_file_path"],
            dirs["workspace_dir"],
            config["language"]
        )
        
        # 5. 数据预处理
        prepare_data(config, dirs, csv_file)
        
        # 6. 模型训练
        run_training(config, dirs)
        
        # 7. 标记训练完成
        logger.complete("succeeded")
        
        print("=" * 60)
        print("训练完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"[FATAL] 训练失败: {str(e)}")
        training_status = "failed"
        logger.complete("failed")
        import traceback
        traceback.print_exc()
    
    # 8. 保存结果
    save_job_result(config, dirs, training_status)
    
    # 9. 回调通知
    notify_server(config, training_status)
    
    sys.exit(0 if training_status == "succeeded" else 1)


if __name__ == "__main__":
    main()
