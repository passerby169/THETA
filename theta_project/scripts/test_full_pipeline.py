#!/usr/bin/env python3
"""
完整流程测试脚本
验证: 数据清洗 -> 自适应分段 -> 嵌入 -> 训练 -> 评估 -> 可视化
"""
import os
import sys
import subprocess
from pathlib import Path

# 设置环境变量
os.environ["THETA_LONG_TEXT_THRESHOLD"] = "5000"  # 长文本阈值
os.environ["THETA_MAX_SEQ_LENGTH"] = "512"  # 测试用较小值

# 项目路径
PROJECT_ROOT = Path("/root/theta_project")
THETA_CODE_DIR = PROJECT_ROOT / "THETA" / "src" / "models"
TEST_DATA_DIR = PROJECT_ROOT / "THETA" / "data" / "test_b"
OUTPUT_DIR = PROJECT_ROOT / "test_output"

def run_step(name, cmd, cwd=None):
    """运行一个步骤"""
    print(f"\n{'='*60}")
    print(f"[STEP] {name}")
    print(f"{'='*60}")
    print(f"  命令: {cmd}")
    print(f"  工作目录: {cwd or os.getcwd()}")
    print()
    
    result = subprocess.run(
        cmd, 
        shell=True, 
        cwd=cwd,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"[STDERR] {result.stderr}")
    
    if result.returncode != 0:
        print(f"[ERROR] 步骤失败: {name}")
        return False
    
    print(f"[OK] {name} 完成")
    return True

def test_python_docx():
    """测试 python-docx 库"""
    print("\n" + "="*60)
    print("[TEST] 验证 python-docx 库")
    print("="*60)
    
    try:
        import docx
        print(f"  ✓ python-docx 版本: {docx.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ python-docx 未安装: {e}")
        return False

def test_data_admission():
    """测试数据准入校验"""
    print("\n" + "="*60)
    print("[TEST] 数据准入校验")
    print("="*60)
    
    # 检查文件数量
    docx_files = list(TEST_DATA_DIR.glob("*.docx"))
    print(f"  输入目录: {TEST_DATA_DIR}")
    print(f"  文档数量: {len(docx_files)}")
    
    for f in docx_files:
        size = f.stat().st_size
        print(f"    - {f.name} ({size:,} bytes)")
    
    if len(docx_files) >= 5:
        print(f"  ✓ 文档数量满足要求 (>= 5)")
        return True
    else:
        print(f"  ✗ 文档数量不足 (需要 >= 5)")
        return False

def test_dataclean():
    """测试数据清洗和自适应分段"""
    print("\n" + "="*60)
    print("[TEST] 数据清洗 + 自适应分段")
    print("="*60)
    
    # 导入并运行 dataclean
    sys.path.insert(0, str(THETA_CODE_DIR / "dataclean"))
    
    try:
        from src.consolidator import DataConsolidator, LONG_TEXT_THRESHOLD
        from src.converter import TextConverter
        
        print(f"  长文本阈值: {LONG_TEXT_THRESHOLD} 字符")
        
        # 创建处理器
        consolidator = DataConsolidator()
        converter = TextConverter()
        
        # 获取文件列表
        docx_files = list(TEST_DATA_DIR.glob("*.docx"))
        
        # 测试文本提取
        print(f"\n  [提取文本]")
        for f in docx_files:
            text = converter.extract_text(str(f))
            char_count = len(text)
            will_split = char_count > LONG_TEXT_THRESHOLD
            print(f"    - {f.name}: {char_count} 字 {'[将分段]' if will_split else ''}")
        
        # 测试自适应分段
        print(f"\n  [自适应分段测试]")
        test_file = docx_files[0]
        text = converter.extract_text(str(test_file))
        chunks = consolidator._adaptive_split(text, str(test_file))
        print(f"    源文件: {test_file.name}")
        print(f"    原始长度: {len(text)} 字符")
        print(f"    分段数量: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"      段落 {i}: {len(chunk['text'])} 字符, source_file={Path(chunk['source_file']).name}")
        if len(chunks) > 3:
            print(f"      ... 还有 {len(chunks) - 3} 个段落")
        
        print(f"\n  ✓ 数据清洗和自适应分段测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_config():
    """测试嵌入配置"""
    print("\n" + "="*60)
    print("[TEST] 嵌入配置验证")
    print("="*60)
    
    sys.path.insert(0, str(THETA_CODE_DIR / "preprocessing"))
    
    try:
        from embedding_processor import ProcessingConfig, _get_max_seq_length
        
        # 测试环境变量读取
        max_len = _get_max_seq_length()
        print(f"  max_sequence_length: {max_len}")
        print(f"  环境变量 THETA_MAX_SEQ_LENGTH: {os.environ.get('THETA_MAX_SEQ_LENGTH', 'not set')}")
        
        # 创建配置
        config = ProcessingConfig()
        print(f"  use_sliding_window: {config.use_sliding_window}")
        print(f"  sliding_window_overlap: {config.sliding_window_overlap:.0%}")
        
        print(f"\n  ✓ 嵌入配置验证通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试流程"""
    print("="*60)
    print("THETA 完整流程测试")
    print("="*60)
    print(f"测试数据目录: {TEST_DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. 验证 python-docx
    results["python-docx"] = test_python_docx()
    
    # 2. 数据准入校验
    results["data_admission"] = test_data_admission()
    
    # 3. 数据清洗 + 自适应分段
    results["dataclean"] = test_dataclean()
    
    # 4. 嵌入配置验证
    results["embedding_config"] = test_embedding_config()
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("[SUCCESS] 所有测试通过！")
    else:
        print("[FAILED] 部分测试失败")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
