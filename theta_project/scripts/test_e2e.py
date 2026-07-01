#!/usr/bin/env python3
"""
端到端测试脚本 - 模拟完整训练流程

流程:
1. 上传测试数据到 OSS raw_data 目录
2. 模拟训练请求 (绕过前端)
3. 提交 DLC 任务
4. 等待任务完成
5. 验证 OSS results 输出
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from utils.oss_util import get_oss_bucket, upload_file_to_oss, sync_theta_project_to_oss
from services.dlc_service import submit_job, get_job_status, generate_run_id


# 测试配置
TEST_USERNAME = "test_user"
TEST_DATASET_NAME = "policy_docs"
TEST_FILE_PATH = "/root/theta_project/test_data/test_data.csv"

# 训练参数
TRAIN_CONFIG = {
    "model_type": "lda",      # 使用 LDA 快速测试
    "model_size": "0.6B",
    "mode": "zero_shot",
    "num_topics": 10,         # 减少主题数加快测试
    "epochs": 50,             # 减少轮数加快测试
    "language": "chinese",
    "vocab_size": 3000,
}


def step1_upload_test_data():
    """步骤 1: 上传测试数据到 OSS"""
    print("=" * 60)
    print("[STEP 1] 上传测试数据到 OSS")
    print("=" * 60)
    
    if not os.path.exists(TEST_FILE_PATH):
        raise FileNotFoundError(f"测试数据不存在: {TEST_FILE_PATH}")
    
    # 上传到 raw_data/{username}/{dataset_name}/
    oss_key = f"raw_data/{TEST_USERNAME}/{TEST_DATASET_NAME}/test_data.csv"
    
    bucket = get_oss_bucket()
    with open(TEST_FILE_PATH, "rb") as f:
        bucket.put_object(oss_key, f)
    
    print(f"  [OK] 已上传: {TEST_FILE_PATH}")
    print(f"  [OK] OSS 路径: oss://{os.getenv('OSS_BUCKET_NAME')}/{oss_key}")
    
    return oss_key


def step2_sync_code():
    """步骤 2: 同步代码到 OSS"""
    print("\n" + "=" * 60)
    print("[STEP 2] 同步 THETA 代码到 OSS")
    print("=" * 60)
    
    result = sync_theta_project_to_oss()
    print(f"  [OK] 同步完成: {result}")
    
    return result


def step3_submit_dlc_job(oss_file_path: str):
    """步骤 3: 提交 DLC 训练任务"""
    print("\n" + "=" * 60)
    print("[STEP 3] 提交 DLC 训练任务")
    print("=" * 60)
    
    # 模拟数据库中的 user_id 和 file_id
    mock_user_id = 1
    mock_file_id = 1
    mock_job_id = int(datetime.now().strftime("%Y%m%d%H%M%S"))
    
    print(f"  [INFO] 用户: {TEST_USERNAME} (id={mock_user_id})")
    print(f"  [INFO] 数据集: {TEST_DATASET_NAME}")
    print(f"  [INFO] 模型: {TRAIN_CONFIG['model_type']}")
    print(f"  [INFO] 主题数: {TRAIN_CONFIG['num_topics']}")
    print(f"  [INFO] 训练轮数: {TRAIN_CONFIG['epochs']}")
    
    try:
        dlc_job_id = submit_job(
            user_id=mock_user_id,
            username=TEST_USERNAME,
            file_id=mock_file_id,
            file_path=oss_file_path,
            job_id=mock_job_id,
            dataset_name=TEST_DATASET_NAME,
            **TRAIN_CONFIG
        )
        
        print(f"  [OK] DLC 任务已提交: {dlc_job_id}")
        return dlc_job_id, mock_job_id
        
    except Exception as e:
        print(f"  [ERROR] 提交失败: {e}")
        raise


def step4_wait_for_completion(dlc_job_id: str, timeout_minutes: int = 30):
    """步骤 4: 等待任务完成"""
    print("\n" + "=" * 60)
    print("[STEP 4] 等待 DLC 任务完成")
    print("=" * 60)
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    poll_interval = 30  # 每 30 秒查询一次
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"  [TIMEOUT] 超过 {timeout_minutes} 分钟，任务可能仍在运行")
            return "timeout"
        
        status = get_job_status(dlc_job_id)
        elapsed_min = int(elapsed / 60)
        elapsed_sec = int(elapsed % 60)
        
        print(f"  [{elapsed_min:02d}:{elapsed_sec:02d}] 状态: {status}")
        
        if status in ["succeeded", "failed"]:
            return status
        
        time.sleep(poll_interval)


def step5_verify_results():
    """步骤 5: 验证 OSS 结果输出"""
    print("\n" + "=" * 60)
    print("[STEP 5] 验证 OSS 结果输出")
    print("=" * 60)
    
    bucket = get_oss_bucket()
    result_prefix = f"results/{TEST_USERNAME}/{TEST_DATASET_NAME}/{TRAIN_CONFIG['model_type']}/"
    
    print(f"  [INFO] 检查路径: oss://{os.getenv('OSS_BUCKET_NAME')}/{result_prefix}")
    
    # 列出结果目录下的文件
    found_files = []
    for obj in bucket.list_objects(prefix=result_prefix).object_list:
        found_files.append(obj.key)
        print(f"  [FILE] {obj.key}")
    
    if found_files:
        print(f"\n  [OK] 找到 {len(found_files)} 个结果文件")
        
        # 检查关键文件
        key_files = ["training_log.json", "job_result.json"]
        for key_file in key_files:
            matching = [f for f in found_files if key_file in f]
            if matching:
                print(f"  [OK] 找到 {key_file}")
            else:
                print(f"  [WARN] 未找到 {key_file}")
        
        return True
    else:
        print("  [WARN] 未找到任何结果文件")
        return False


def main():
    print("\n" + "=" * 60)
    print("THETA 端到端测试")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试数据: {TEST_FILE_PATH}")
    print(f"目标用户: {TEST_USERNAME}")
    print(f"数据集名: {TEST_DATASET_NAME}")
    
    try:
        # Step 1: 上传测试数据
        oss_file_path = step1_upload_test_data()
        
        # Step 2: 同步代码
        step2_sync_code()
        
        # Step 3: 提交 DLC 任务
        dlc_job_id, job_id = step3_submit_dlc_job(oss_file_path)
        
        print("\n" + "=" * 60)
        print("DLC 任务已提交!")
        print("=" * 60)
        print(f"DLC Job ID: {dlc_job_id}")
        print(f"Local Job ID: {job_id}")
        print(f"\n可以在阿里云 PAI-DLC 控制台查看任务状态")
        print(f"或运行以下命令查询状态:")
        print(f"  python -c \"from services.dlc_service import get_job_status; print(get_job_status('{dlc_job_id}'))\"")
        
        # 询问是否等待完成
        print("\n是否等待任务完成? (y/n): ", end="")
        wait_input = input().strip().lower()
        
        if wait_input == 'y':
            # Step 4: 等待完成
            final_status = step4_wait_for_completion(dlc_job_id)
            
            if final_status == "succeeded":
                # Step 5: 验证结果
                step5_verify_results()
        
        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FATAL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
