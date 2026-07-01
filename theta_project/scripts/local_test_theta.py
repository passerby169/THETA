#!/usr/bin/env python3
"""
本地测试脚本 - THETA 模型 (模拟 Qwen 嵌入)
由于本地没有 GPU 和 Qwen 模型，使用随机嵌入模拟

测试数据集: 宁夏回族自治区
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import json

# =============================================================================
# 配置
# =============================================================================
USERNAME = "test_user"
DATASET_NAME = "ningxia"
MODEL_TYPE = "theta"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

INPUT_DIR = f"/mnt/raw_data/{USERNAME}/{DATASET_NAME}"
OUTPUT_DIR = f"/mnt/results/{USERNAME}/{DATASET_NAME}/{MODEL_TYPE}/{RUN_ID}"
WORKSPACE_DIR = f"{OUTPUT_DIR}/workspace"
MODEL_DIR = f"{OUTPUT_DIR}/model"
EVALUATION_DIR = f"{OUTPUT_DIR}/evaluation"
VISUALIZATION_DIR = f"{OUTPUT_DIR}/visualization"

print("=" * 60)
print("本地测试流程 - THETA 模型 (模拟 Qwen 嵌入)")
print("=" * 60)
print(f"用户: {USERNAME}")
print(f"数据集: {DATASET_NAME}")
print(f"模型: {MODEL_TYPE}")
print(f"输出目录: {OUTPUT_DIR}")
print()

# =============================================================================
# Step 1: 创建目录
# =============================================================================
print("=== Step 1: 创建目录结构 ===")
for d in [WORKSPACE_DIR, MODEL_DIR, EVALUATION_DIR, VISUALIZATION_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"  Created: {d}")

# =============================================================================
# Step 2: 复用 LDA 的 BOW 数据
# =============================================================================
print("\n=== Step 2: 复用 BOW 数据 ===")

# 查找最新的 LDA 结果
lda_results = sorted(Path(f"/mnt/results/{USERNAME}/{DATASET_NAME}/lda").glob("*"))
if not lda_results:
    print("[ERROR] No LDA results found. Run LDA first.")
    sys.exit(1)

latest_lda = lda_results[-1]
lda_workspace = latest_lda / "workspace"

import shutil
for f in ["bow_matrix.npy", "vocab.json", "data.csv"]:
    src = lda_workspace / f
    dst = Path(WORKSPACE_DIR) / f
    if src.exists():
        shutil.copy(src, dst)
        print(f"  Copied: {f}")

# =============================================================================
# Step 3: 生成模拟 Qwen 嵌入 (随机向量)
# =============================================================================
print("\n=== Step 3: 生成模拟 Qwen 嵌入 ===")

bow_matrix = np.load(Path(WORKSPACE_DIR) / "bow_matrix.npy")
with open(Path(WORKSPACE_DIR) / "vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

n_docs, vocab_size = bow_matrix.shape
embedding_dim = 1024  # Qwen 0.6B 维度

# 模拟文档嵌入 (实际应由 Qwen 生成)
print(f"  Generating mock document embeddings: ({n_docs}, {embedding_dim})")
doc_embeddings = np.random.randn(n_docs, embedding_dim).astype(np.float32)
doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

# 模拟词表嵌入 (实际应由 Qwen 生成)
print(f"  Generating mock vocab embeddings: ({vocab_size}, {embedding_dim})")
vocab_embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
vocab_embeddings = vocab_embeddings / np.linalg.norm(vocab_embeddings, axis=1, keepdims=True)

# 保存
np.save(Path(WORKSPACE_DIR) / f"{DATASET_NAME}_zero_shot_embeddings.npy", doc_embeddings)
np.save(Path(WORKSPACE_DIR) / "embeddings.npy", doc_embeddings)
np.save(Path(WORKSPACE_DIR) / "vocab_embeddings.npy", vocab_embeddings)

print(f"  Saved to: {WORKSPACE_DIR}")

# =============================================================================
# Step 4: THETA 模型训练 (简化版 VAE)
# =============================================================================
print("\n=== Step 4: THETA 模型训练 ===")

# 由于没有完整的 THETA 实现，这里使用简化的 NMF + 嵌入融合
from sklearn.decomposition import NMF

n_topics = 10
print(f"  Training THETA (NMF + Embedding fusion) with {n_topics} topics...")

# 使用 NMF 作为基础
nmf = NMF(n_components=n_topics, max_iter=200, random_state=42)
theta = nmf.fit_transform(bow_matrix)  # 文档-主题分布
beta = nmf.components_  # 主题-词分布

# 归一化
theta = theta / (theta.sum(axis=1, keepdims=True) + 1e-10)
beta = beta / (beta.sum(axis=1, keepdims=True) + 1e-10)

# 融合嵌入信息 (简化版)
# 实际 THETA 使用 VAE + Qwen 嵌入，这里只是模拟
topic_embeddings = np.dot(beta, vocab_embeddings)  # 主题嵌入

# 保存
model_output_dir = Path(MODEL_DIR) / "theta"
model_output_dir.mkdir(parents=True, exist_ok=True)

np.save(model_output_dir / f"theta_k{n_topics}.npy", theta)
np.save(model_output_dir / f"beta_k{n_topics}.npy", beta)
np.save(model_output_dir / f"topic_embeddings_k{n_topics}.npy", topic_embeddings)

# 提取主题词
topic_words = []
for topic_idx, topic in enumerate(beta):
    top_word_indices = topic.argsort()[-20:][::-1]
    top_words = [vocab[i] for i in top_word_indices]
    topic_words.append(top_words)
    # 过滤标点
    filtered = [w for w in top_words if len(w) > 1 and not any(c in w for c in '，。、；：""''（）【】《》/._\n\r\t ')][:5]
    print(f"  Topic {topic_idx}: {', '.join(filtered)}...")

with open(model_output_dir / f"topic_words_k{n_topics}.json", "w", encoding="utf-8") as f:
    json.dump(topic_words, f, ensure_ascii=False, indent=2)

print(f"\n  Theta shape: {theta.shape}")
print(f"  Beta shape: {beta.shape}")
print(f"  Topic embeddings shape: {topic_embeddings.shape}")
print("\n[OK] THETA Training completed")

# =============================================================================
# Step 5: 生成可视化
# =============================================================================
print("\n=== Step 5: 生成可视化 ===")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

vis_dir = Path(VISUALIZATION_DIR)
vis_dir.mkdir(parents=True, exist_ok=True)

# 查找中文字体
font_path = None
for fp in ['/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
           '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
           '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if Path(fp).exists():
        font_path = fp
        break

# 生成词云
print("  Generating word clouds...")
for topic_idx, words in enumerate(topic_words):
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
    plt.title(f'THETA Topic {topic_idx}')
    plt.savefig(vis_dir / f'wordcloud_topic_{topic_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"  Generated {len(topic_words)} word cloud images")

# 主题分布图
print("  Generating topic distribution...")
topic_dist = theta.mean(axis=0)

plt.figure(figsize=(12, 6))
plt.bar(range(len(topic_dist)), topic_dist, color='coral')
plt.xlabel('Topic')
plt.ylabel('Average Probability')
plt.title('THETA Topic Distribution')
plt.savefig(vis_dir / 'topic_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 主题嵌入 t-SNE 可视化
print("  Generating topic embedding visualization...")
try:
    from sklearn.manifold import TSNE
    
    # 合并文档嵌入和主题嵌入进行可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, n_topics-1))
    topic_2d = tsne.fit_transform(topic_embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(topic_2d[:, 0], topic_2d[:, 1], c=range(n_topics), cmap='tab10', s=200)
    for i, (x, y) in enumerate(topic_2d):
        plt.annotate(f'T{i}', (x, y), fontsize=12, ha='center', va='bottom')
    plt.title('THETA Topic Embeddings (t-SNE)')
    plt.savefig(vis_dir / 'topic_embeddings_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Generated topic_embeddings_tsne.png")
except Exception as e:
    print(f"  [WARN] t-SNE visualization failed: {e}")

print(f"  Saved visualizations to: {vis_dir}")

# =============================================================================
# 输出目录结构
# =============================================================================
print("\n=== 输出目录结构 ===")
for root, dirs, files in os.walk(OUTPUT_DIR):
    level = root.replace(OUTPUT_DIR, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:10]:
        print(f'{subindent}{file}')
    if len(files) > 10:
        print(f'{subindent}... and {len(files) - 10} more files')

print("\n" + "=" * 60)
print("THETA 测试完成!")
print("=" * 60)
