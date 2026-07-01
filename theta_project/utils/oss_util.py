"""
OSS 工具模块 - 阿里云 OSS 文件操作

功能:
1. 上传/下载文件（支持多线程并发上传）
2. 同步代码目录到 OSS（多线程并发上传）
3. 同步 THETA 项目到 OSS（并发执行所有上传任务）
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import oss2
from dotenv import load_dotenv

load_dotenv()

ALIBABA_CLOUD_ACCESS_KEY_ID = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")
ALIBABA_CLOUD_ACCESS_KEY_SECRET = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
OSS_BUCKET_NAME = os.getenv("OSS_BUCKET_NAME")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 并发上传线程数
UPLOAD_WORKERS = int(os.getenv("OSS_UPLOAD_WORKERS", "8"))

# ---------------------------------------------------------------------------
# Bucket Client（每次请求创建新的 client 避免连接复用问题）
# ---------------------------------------------------------------------------

def get_oss_bucket() -> oss2.Bucket:
    """Create and return OSS bucket client."""
    auth = oss2.Auth(ALIBABA_CLOUD_ACCESS_KEY_ID, ALIBABA_CLOUD_ACCESS_KEY_SECRET)
    return oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)


# ---------------------------------------------------------------------------
# 单文件上传/下载
# ---------------------------------------------------------------------------

def upload_file_to_oss(local_path: str, oss_key: str) -> str:
    """
    Upload any file to OSS.

    Returns:
        The OSS URL of the uploaded file
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File not found: {local_path}")

    bucket = get_oss_bucket()
    with open(local_path, "rb") as f:
        bucket.put_object(oss_key, f)

    oss_url = f"oss://{OSS_BUCKET_NAME}/{oss_key}"
    print(f"[INFO] Uploaded {local_path} to {oss_url}")
    return oss_url


def download_file_from_oss(oss_key: str, local_path: str) -> str:
    """Download a file from OSS."""
    bucket = get_oss_bucket()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    bucket.get_object_to_file(oss_key, local_path)
    print(f"[INFO] Downloaded oss://{OSS_BUCKET_NAME}/{oss_key} to {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# 多线程并发上传（核心优化）
# ---------------------------------------------------------------------------

def _upload_single_file(oss_key: str, local_path: str) -> bool:
    """
    上传单个文件（供 ThreadPoolExecutor 调用）。
    成功返回 True，失败返回 False。
    """
    try:
        with open(local_path, "rb") as f:
            get_oss_bucket().put_object(oss_key, f)
        return True
    except Exception as e:
        print(f"[WARN] Failed to upload {local_path}: {e}")
        return False


def sync_directory_to_oss(
    local_dir: str,
    oss_prefix: str,
    exclude_patterns: Optional[list] = None,
    max_workers: Optional[int] = None,
) -> int:
    """
    同步本地目录到 OSS（多线程并发上传）。

    对于包含 N 个文件的目录，使用 max_workers 个线程并发上传，
    将串行上传耗时从 O(N * t_upload) 降低到 O((N/max_workers) * t_upload)。

    Parameters
    ----------
    local_dir : str
        本地目录路径
    oss_prefix : str
        OSS 前缀路径 (如 "code/src/models/")
    exclude_patterns : list | None
        排除的文件/目录模式列表
    max_workers : int | None
        并发线程数，默认取环境变量 OSS_UPLOAD_WORKERS（默认 8）

    Returns
    -------
    int
        成功上传的文件数量
    """
    if exclude_patterns is None:
        exclude_patterns = [
            "__pycache__", ".git", ".pyc", ".pyo", ".egg-info",
            ".DS_Store", "*.log", "*.tmp",
        ]

    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Directory not found: {local_dir}")

    # 收集所有待上传文件
    tasks: list[tuple[str, str]] = []  # [(oss_key, local_path), ...]
    for file_path in local_path.rglob("*"):
        if file_path.is_dir():
            continue
        relative_path = file_path.relative_to(local_path)
        skip = any(p in str(relative_path) for p in exclude_patterns)
        if skip:
            continue
        oss_key = f"{oss_prefix.rstrip('/')}/{relative_path}"
        tasks.append((oss_key, str(file_path)))

    if not tasks:
        return 0

    workers = max_workers or UPLOAD_WORKERS
    uploaded = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_upload_single_file, oss_key, lp): (oss_key, lp)
            for oss_key, lp in tasks
        }
        for future in as_completed(futures):
            if future.result():
                uploaded += 1

    print(
        f"[INFO] Synced {uploaded}/{len(tasks)} files from {local_dir} "
        f"to oss://{OSS_BUCKET_NAME}/{oss_prefix} ({workers} workers)"
    )
    return uploaded


# ---------------------------------------------------------------------------
# THETA 项目并发同步（最高优先级优化）
# ---------------------------------------------------------------------------

def sync_theta_project_to_oss() -> dict:
    """
    同步 THETA 项目到 OSS（所有上传任务并发执行）。

    上传结构:
        code/dlc_entry.py              <- dlc_deployment/dlc_entry.py
        code/scripts/                  <- THETA/scripts/
        code/src/models/              <- THETA/src/models/
        code/services/dlc_service.py   <- services/dlc_service.py

    Returns
    -------
    dict
        {"dlc_entry": 1, "src_models": count, "theta_scripts": count, "dlc_service": 1}
    """
    result: dict = {}
    workers = UPLOAD_WORKERS

    # 每个上传任务: (name, callable, args)
    upload_tasks: list[tuple[str, callable, tuple]] = []

    # 1. dlc_entry.py
    dlc_entry_path = PROJECT_ROOT / "dlc_deployment" / "dlc_entry.py"
    if dlc_entry_path.exists():
        def _upload_entry():
            with open(dlc_entry_path, "rb") as f:
                get_oss_bucket().put_object("code/dlc_entry.py", f)
            print(f"[INFO] Synced dlc_entry.py -> oss://{OSS_BUCKET_NAME}/code/dlc_entry.py")
            return 1
        upload_tasks.append(("dlc_entry", _upload_entry, ()))

    # 2. src/models/ 目录（调用已有的多线程版本）
    theta_src_models = PROJECT_ROOT / "THETA" / "src" / "models"
    if theta_src_models.exists():
        def _upload_models():
            return sync_directory_to_oss(
                str(theta_src_models),
                "code/src/models/",
                exclude_patterns=["__pycache__", ".pyc", "logs/", "*.log", ".git"],
                max_workers=workers,
            )
        upload_tasks.append(("src_models", _upload_models, ()))

    # 3. scripts/ 目录
    theta_scripts = PROJECT_ROOT / "THETA" / "scripts"
    if theta_scripts.exists():
        def _upload_scripts():
            return sync_directory_to_oss(
                str(theta_scripts),
                "code/scripts/",
                exclude_patterns=["__pycache__", ".pyc", "*.log"],
                max_workers=workers,
            )
        upload_tasks.append(("theta_scripts", _upload_scripts, ()))

    # 4. dlc_service.py
    dlc_service_path = PROJECT_ROOT / "services" / "dlc_service.py"
    if dlc_service_path.exists():
        def _upload_service():
            with open(dlc_service_path, "rb") as f:
                get_oss_bucket().put_object("code/services/dlc_service.py", f)
            print(f"[INFO] Synced dlc_service.py -> oss://{OSS_BUCKET_NAME}/code/services/dlc_service.py")
            return 1
        upload_tasks.append(("dlc_service", _upload_service, ()))

    # 并发执行所有上传任务
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(func, *args): name
            for name, func, args in upload_tasks
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result[name] = future.result()
            except Exception as e:
                print(f"[WARN] sync_theta_project_to_oss task '{name}' failed: {e}")
                result[name] = 0

    print(f"[INFO] THETA project sync complete: {result}")
    return result


# ---------------------------------------------------------------------------
# 遗留函数（保持兼容）
# ---------------------------------------------------------------------------

def upload_train_script(local_path: str = None, oss_key: str = "scripts/train_engine.py") -> str:
    """Legacy: 上传训练脚本到 OSS（单文件，使用并发上传路径）。"""
    if local_path is None:
        local_path = str(PROJECT_ROOT / "scripts" / "train_engine.py")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Script not found: {local_path}")
    return upload_file_to_oss(local_path, oss_key)


def sync_code_to_oss() -> str:
    """Legacy: 同步 train_engine.py（已被 sync_theta_project_to_oss 取代）。"""
    return upload_train_script()
