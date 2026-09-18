import os
import numpy as np
from PIL import Image
import torch
import clip
from tqdm import tqdm
import h5py
from pathlib import Path
from collections import Counter

# ==================== 配置区域（请修改这里） ====================
ROOT = r"D:\PycharmProject\xixi\MIRFlickr-25k\mirflickr"   # 改成你实际的 mirflickr 路径
OUTPUT_DIR = r"C:\Users\Administrator\Desktop\gitdemo\Dataset\flickr25k"  # 输出目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_NUM = 5000
QUERY_NUM = 2000
# ==============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载 CLIP
print("Loading CLIP ViT-B/32 ...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

# 1. 收集所有有效样本（图片 + tags）
img_dir = Path(ROOT) / "images"
tag_dir = Path(ROOT) / "meta" / "tags"
ann_dir = Path(ROOT) / "annotations"

# 获取所有图片 id（1~25000）
all_ids = []
for i in range(1, 25001):
    img_path = img_dir / f"im{i}.jpg"
    tag_path = tag_dir / f"tags{i}.txt"
    if img_path.exists() and tag_path.exists():
        all_ids.append(i)

print(f"找到有效样本数量: {len(all_ids)}")

# 2. 读取 24 类标签
# 优先使用不带 _r1 的文件
label_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.txt') and not f.endswith('_r1.txt') and f != 'README.txt'])
print(f"找到标签文件数量: {len(label_files)}")
print("标签类别:", [f.replace('.txt','') for f in label_files])

# 构建 label 矩阵 (N, 24)
N = len(all_ids)
id_to_idx = {img_id: idx for idx, img_id in enumerate(all_ids)}
labels = np.zeros((N, len(label_files)), dtype=np.float32)

for c, fname in enumerate(label_files):
    with open(ann_dir / fname, 'r') as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                img_id = int(line)
                if img_id in id_to_idx:
                    labels[id_to_idx[img_id], c] = 1.0

# ==================== 过滤文本和标签为0的样本 ====================
print("正在统计全数据集 tag 出现频次...")
all_tags_list = []
tag_counts_per_sample = []

for img_id in all_ids:
    tag_path = tag_dir / f"tags{img_id}.txt"
    with open(tag_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().strip().replace('\n', ' ')
        tags = content.split()
    all_tags_list.extend(tags)
    tag_counts_per_sample.append(len(tags))

# 全局频次统计
tag_freq = Counter(all_tags_list)
print(f"原始不同 tag 数量: {len(tag_freq)}")

# 只保留出现次数 ≥ 20 的 tag
frequent_tags = {tag for tag, cnt in tag_freq.items() if cnt >= 20}
print(f"出现频次 ≥ 20 的 tag 数量: {len(frequent_tags)}")

# 重新统计每个样本在「高频 tag」下的有效 tag 数量
valid_tag_counts = []
for img_id in all_ids:
    tag_path = tag_dir / f"tags{img_id}.txt"
    with open(tag_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().strip().replace('\n', ' ')
        tags = content.split()
    # 只计算属于高频词的 tag
    valid_tags = [t for t in tags if t in frequent_tags]
    valid_tag_counts.append(len(valid_tags))

valid_tag_counts = np.array(valid_tag_counts)
label_counts = labels.sum(axis=1)

print(f"有效 tag 数量统计: 最小={valid_tag_counts.min()}, 最大={valid_tag_counts.max()}, 平均={valid_tag_counts.mean():.1f}")

# 最终过滤条件：
# 1. 标签数量 > 0
# 2. 有效高频 tag 数量 > 0（也可以改成 >=1 或更高）
valid_mask = (label_counts > 0) & (valid_tag_counts > 0)

print(f"原始样本数: {len(all_ids)}")
print(f"标签为0的样本数: {(label_counts == 0).sum()}")
print(f"有效高频tag为0的样本数: {(valid_tag_counts == 0).sum()}")
print(f"过滤后有效样本数量: {valid_mask.sum()}")

# 应用过滤
all_ids = [all_ids[i] for i in range(len(all_ids)) if valid_mask[i]]
labels = labels[valid_mask]
print(f"最终保留样本数: {len(all_ids)}")
# ================================================================

# 3. 随机划分（固定随机种子保证可复现）
np.random.seed(42)
indices = np.random.permutation(len(all_ids))

query_idx = indices[:QUERY_NUM]
train_idx = indices[QUERY_NUM:QUERY_NUM + TRAIN_NUM]
db_idx    = indices[QUERY_NUM:]          # 检索库 = 除 query 外的所有

print(f"Train: {len(train_idx)}, Query: {len(query_idx)}, DB: {len(db_idx)}")

# 4. 提取特征函数
def extract_image_features(ids):
    feats = []
    with torch.no_grad():
        for img_id in tqdm(ids, desc="Extracting image features"):
            img_path = img_dir / f"im{img_id}.jpg"
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)

def extract_text_features(ids):
    feats = []
    with torch.no_grad():
        for img_id in tqdm(ids, desc="Extracting text features"):
            tag_path = tag_dir / f"tags{img_id}.txt"
            with open(tag_path, 'r', encoding='utf-8', errors='ignore') as f:
                tags = f.read().strip().replace('\n', ' ')
            if not tags:
                tags = "photo"          # 防止空文本
            text = clip.tokenize([tags], truncate=True).to(DEVICE)
            feat = model.encode_text(text)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)

# 5. 开始提取
print("\n===== 提取 Query 特征 =====")
I_te = extract_image_features([all_ids[i] for i in query_idx])
T_te = extract_text_features([all_ids[i] for i in query_idx])
L_te = labels[query_idx]

print("\n===== 提取 Train 特征 =====")
I_tr = extract_image_features([all_ids[i] for i in train_idx])
T_tr = extract_text_features([all_ids[i] for i in train_idx])
L_tr = labels[train_idx]

print("\n===== 提取 Database 特征 =====")
I_db = extract_image_features([all_ids[i] for i in db_idx])
T_db = extract_text_features([all_ids[i] for i in db_idx])
L_db = labels[db_idx]

# 6. 保存为 CDTH 需要的格式
save_path = os.path.join(OUTPUT_DIR, "flickr25k_clip_split.mat")
print(f"\n正在保存到: {save_path}")

with h5py.File(save_path, 'w') as f:
    f.create_dataset('I_tr', data=I_tr.T)   # 注意：CDTH 代码里会再 .T，所以这里先转置
    f.create_dataset('T_tr', data=T_tr.T)
    f.create_dataset('L_tr', data=L_tr.T)
    f.create_dataset('I_te', data=I_te.T)
    f.create_dataset('T_te', data=T_te.T)
    f.create_dataset('L_te', data=L_te.T)
    f.create_dataset('I_db', data=I_db.T)
    f.create_dataset('T_db', data=T_db.T)
    f.create_dataset('L_db', data=L_db.T)

print("保存完成！")
print(f"I_tr shape: {I_tr.shape}, T_tr shape: {T_tr.shape}, L_tr shape: {L_tr.shape}")
print(f"I_te shape: {I_te.shape}, T_te shape: {T_te.shape}")
print(f"I_db shape: {I_db.shape}")