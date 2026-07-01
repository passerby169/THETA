#!/usr/bin/env python3
"""
本地测试脚本 - 模拟 DLC 容器完整流程
不使用 DLC，但目录结构和调用逻辑完全一致

测试数据集: 宁夏回族自治区 (41 个 doc/docx 文件)
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 禁用 bootstrap_environment 中的自动安装
os.environ["SKIP_DEPENDENCY_CHECK"] = "1"

# =============================================================================
# 配置 (模拟 DLC 环境变量)
# =============================================================================
USERNAME = "test_user"
DATASET_NAME = "ningxia"
MODEL_TYPE = "lda"  # LDA 不需要 torch
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# 模拟 DLC 挂载点
INPUT_DIR = f"/mnt/raw_data/{USERNAME}/{DATASET_NAME}"
OUTPUT_DIR = f"/mnt/results/{USERNAME}/{DATASET_NAME}/{MODEL_TYPE}/{RUN_ID}"
WORKSPACE_DIR = f"{OUTPUT_DIR}/workspace"
MODEL_DIR = f"{OUTPUT_DIR}/model"
EVALUATION_DIR = f"{OUTPUT_DIR}/evaluation"
VISUALIZATION_DIR = f"{OUTPUT_DIR}/visualization"

# 本地代码路径 (模拟 /mnt/code)
CODE_DIR = "/root/theta_project/THETA/src/models"

# 设置环境变量
os.environ["USERNAME"] = USERNAME
os.environ["DATASET_NAME"] = DATASET_NAME
os.environ["MODEL_TYPE"] = MODEL_TYPE
os.environ["RUN_ID"] = RUN_ID
os.environ["INPUT_DIR"] = INPUT_DIR
os.environ["OUTPUT_DIR"] = OUTPUT_DIR
os.environ["WORKSPACE_DIR"] = WORKSPACE_DIR
os.environ["MODEL_DIR"] = MODEL_DIR
os.environ["EVALUATION_DIR"] = EVALUATION_DIR
os.environ["VISUALIZATION_DIR"] = VISUALIZATION_DIR
os.environ["NUM_TOPICS"] = "10"
os.environ["EPOCHS"] = "50"
os.environ["LANGUAGE"] = "chinese"
os.environ["VOCAB_SIZE"] = "3000"
os.environ["MODE"] = "zero_shot"
os.environ["MODEL_SIZE"] = "0.6B"

# 注入代码路径
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, str(Path(CODE_DIR).parent.parent))

print("=" * 60)
print("本地测试流程 - 模拟 DLC 容器")
print("=" * 60)
print(f"用户: {USERNAME}")
print(f"数据集: {DATASET_NAME}")
print(f"模型: {MODEL_TYPE}")
print(f"输入目录: {INPUT_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print()

# =============================================================================
# Step 1: 创建目录结构
# =============================================================================
print("=== Step 1: 创建目录结构 ===")
for d in [WORKSPACE_DIR, MODEL_DIR, EVALUATION_DIR, VISUALIZATION_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"  Created: {d}")

# =============================================================================
# Step 2: 文档转换 (doc/docx -> CSV)
# =============================================================================
print("\n=== Step 2: 文档转换 ===")

import csv

def convert_docs_to_csv(input_dir: str, output_csv: str):
    """将 doc/docx 文件转换为 CSV"""
    from pathlib import Path
    
    # 尝试导入文档处理库
    try:
        import docx
        has_docx = True
    except ImportError:
        print("  [WARN] python-docx not installed, installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "python-docx", "-q"])
        import docx
        has_docx = True
    
    docs = []
    doc_id = 1
    
    # 递归查找所有 doc/docx 文件
    input_path = Path(input_dir)
    for file_path in input_path.rglob("*"):
        if file_path.suffix.lower() in ['.doc', '.docx']:
            try:
                if file_path.suffix.lower() == '.docx':
                    doc = docx.Document(str(file_path))
                    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                else:
                    # .doc 文件需要特殊处理，尝试用 antiword 或跳过
                    try:
                        result = subprocess.run(
                            ["antiword", str(file_path)],
                            capture_output=True, text=True, timeout=30
                        )
                        text = result.stdout
                    except:
                        print(f"  [SKIP] Cannot read .doc file: {file_path.name}")
                        continue
                
                if text.strip():
                    # 提取年份作为元数据
                    year = ""
                    for part in file_path.parts:
                        if "年" in part:
                            year = part.replace("年", "")
                            break
                    
                    docs.append({
                        "id": doc_id,
                        "text": text.strip(),
                        "source_file": file_path.name,
                        "year": year
                    })
                    doc_id += 1
                    print(f"  [OK] {file_path.name} ({len(text)} chars)")
            except Exception as e:
                print(f"  [ERROR] {file_path.name}: {e}")
    
    # 写入 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "source_file", "year"])
        writer.writeheader()
        writer.writerows(docs)
    
    print(f"\n  Total: {len(docs)} documents converted")
    print(f"  Output: {output_csv}")
    return len(docs)

# 检查是否已有 CSV
data_csv = Path(INPUT_DIR) / "data.csv"
if not data_csv.exists():
    doc_count = convert_docs_to_csv(INPUT_DIR, str(data_csv))
else:
    print(f"  Using existing CSV: {data_csv}")
    import pandas as pd
    df = pd.read_csv(data_csv)
    doc_count = len(df)
    print(f"  Documents: {doc_count}")

# =============================================================================
# Step 3: 数据预处理 (BOW 生成 - LDA 只需要 BOW)
# =============================================================================
print("\n=== Step 3: 数据预处理 (BOW) ===")

import shutil
import pandas as pd
import numpy as np
import json

# 复制 CSV 到 workspace
workspace_csv = Path(WORKSPACE_DIR) / "data.csv"
shutil.copy(data_csv, workspace_csv)

# 直接生成 BOW (不依赖 prepare_data.py 避免 torch 依赖)
print("  Generating BOW matrix...")

df = pd.read_csv(workspace_csv)
texts = df['text'].tolist()

# 中文分词
try:
    import jieba
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "jieba", "-q"])
    import jieba

from sklearn.feature_extraction.text import CountVectorizer

# 分词函数
def chinese_tokenizer(text):
    return list(jieba.cut(text))

# 构建词汇表和 BOW
vectorizer = CountVectorizer(
    tokenizer=chinese_tokenizer,
    max_features=3000,
    min_df=2,
    max_df=0.9,
)

bow_matrix = vectorizer.fit_transform(texts)
vocab = vectorizer.get_feature_names_out().tolist()

# 保存
np.save(Path(WORKSPACE_DIR) / "bow_matrix.npy", bow_matrix.toarray())
with open(Path(WORKSPACE_DIR) / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"  BOW matrix shape: {bow_matrix.shape}")
print(f"  Vocab size: {len(vocab)}")
print(f"  Saved to: {WORKSPACE_DIR}")

print("\n[OK] Data preparation completed")

# =============================================================================
# Step 4: LDA 模型训练 (直接调用 sklearn)
# =============================================================================
print(f"\n=== Step 4: {MODEL_TYPE.upper()} 模型训练 ===")

from sklearn.decomposition import LatentDirichletAllocation

# 加载 BOW
bow_matrix = np.load(Path(WORKSPACE_DIR) / "bow_matrix.npy")
with open(Path(WORKSPACE_DIR) / "vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

print(f"  BOW shape: {bow_matrix.shape}")
print(f"  Training LDA with 10 topics...")

# 训练 LDA
lda = LatentDirichletAllocation(
    n_components=10,
    max_iter=50,
    learning_method='online',
    random_state=42,
    verbose=1,
)
theta = lda.fit_transform(bow_matrix)  # 文档-主题分布
beta = lda.components_  # 主题-词分布
beta = beta / beta.sum(axis=1, keepdims=True)  # 归一化

# 保存模型输出
model_output_dir = Path(MODEL_DIR) / "lda"
model_output_dir.mkdir(parents=True, exist_ok=True)

np.save(model_output_dir / "theta_k10.npy", theta)
np.save(model_output_dir / "beta_k10.npy", beta)

# 提取主题词
topic_words = []
for topic_idx, topic in enumerate(beta):
    top_word_indices = topic.argsort()[-20:][::-1]
    top_words = [vocab[i] for i in top_word_indices]
    topic_words.append(top_words)
    print(f"  Topic {topic_idx}: {', '.join(top_words[:5])}...")

with open(model_output_dir / "topic_words_k10.json", "w", encoding="utf-8") as f:
    json.dump(topic_words, f, ensure_ascii=False, indent=2)

print(f"\n  Theta shape: {theta.shape}")
print(f"  Beta shape: {beta.shape}")
print(f"  Saved to: {model_output_dir}")
print("\n[OK] LDA Training completed")

# =============================================================================
# Step 5: 生成可视化
# =============================================================================
print(f"\n=== Step 5: 生成可视化 ===")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "wordcloud", "matplotlib", "-q"])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

vis_dir = Path(VISUALIZATION_DIR)
vis_dir.mkdir(parents=True, exist_ok=True)

# 生成词云
print("  Generating word clouds...")
font_path = None
for fp in ['/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
           '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
           '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if Path(fp).exists():
        font_path = fp
        break

for topic_idx, words in enumerate(topic_words):
    # 过滤掉标点符号和特殊字符
    filtered_words = [w for w in words if len(w) > 1 and not any(c in w for c in '，。、；：""''（）【】《》/._\n\r\t ')]
    if not filtered_words:
        filtered_words = ["empty"]
    word_freq = {w: 1.0/(i+1) for i, w in enumerate(filtered_words[:20])}
    
    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        font_path=font_path
    ).generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Topic {topic_idx}')
    plt.savefig(vis_dir / f'wordcloud_topic_{topic_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"  Generated {len(topic_words)} word cloud images")

# 生成主题分布图
print("  Generating topic distribution...")
topic_dist = theta.mean(axis=0)

plt.figure(figsize=(12, 6))
plt.bar(range(len(topic_dist)), topic_dist, color='steelblue')
plt.xlabel('Topic')
plt.ylabel('Average Probability')
plt.title('Topic Distribution')
plt.savefig(vis_dir / 'topic_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved visualizations to: {vis_dir}")

# 检查输出文件
print("\n=== 输出目录结构 ===")
for root, dirs, files in os.walk(OUTPUT_DIR):
    level = root.replace(OUTPUT_DIR, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:10]:  # 最多显示 10 个文件
        print(f'{subindent}{file}')
    if len(files) > 10:
        print(f'{subindent}... and {len(files) - 10} more files')

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
