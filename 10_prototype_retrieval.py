import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径配置
# ============================================================
path_data = "./data/test7"
path_label = "./data/test1"

TRAIN_EMB_FILE   = os.path.join(path_data, "training_data_embedding.txt")
TEST_EMB_FILE    = os.path.join(path_data, "testing_data_embedding.txt")
TRAIN_LABEL_FILE = os.path.join(path_label, "training_label.txt")

TRAIN_DATA_TXT   = os.path.join(path_label, "training_data.txt")
TEST_DATA_TXT    = os.path.join(path_label, "testing_data.txt")

MOLECULE_TABLE   = os.path.join(path_label, "Core_Components_reordered.txt")

OUTPUT_DIR = "./data/test10"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 工具函数
# ============================================================
def read_label(file_path):
    vals = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace("*10^", "e")
            vals.append(float(line))
    return np.array(vals)


def read_selfies_from_last_column(file_path):
    selfies_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            selfies_list.append(parts[-1])
    return selfies_list


def build_selfies_to_name_map(table_path):
    mapping = {}
    with open(table_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) <= 9:
                continue
            selfies = parts[7]     # 第 7 列
            name = parts[9]        # 第 9 列
            if selfies not in mapping:
                mapping[selfies] = name
    return mapping


def lookup_name_by_selfies(selfies, s2n_map):
    return s2n_map.get(selfies, "UNKNOWN")


# ============================================================
# 3. 加载数据
# ============================================================
print("\n=== Loading embeddings and labels ===")
X_train = np.loadtxt(TRAIN_EMB_FILE)
X_test  = np.loadtxt(TEST_EMB_FILE)

y_train = read_label(TRAIN_LABEL_FILE)
y_train_log = np.log10(y_train)

train_selfies = read_selfies_from_last_column(TRAIN_DATA_TXT)
test_selfies  = read_selfies_from_last_column(TEST_DATA_TXT)

selfies_to_name = build_selfies_to_name_map(MOLECULE_TABLE)

print(f"Train samples = {len(X_train)}, Test samples = {len(X_test)}")


# ============================================================
# 3.5 结构增强补丁
# ============================================================
print("\n=== Enhancing SELFIES embedding weight ===")
D = 32         # 每个变量维度
W = 3.0        # SELFIES 权重倍率
selfies_slice = slice(5 * D, 6 * D)   # 第 6 个变量（SELFIES）

print(f"Applying weight {W} to SELFIES slice {selfies_slice}")

X_train[:, selfies_slice] *= W
X_test[:, selfies_slice]  *= W


# ============================================================
# 4. KMeans 聚类
# ============================================================
n_clusters = 8
print(f"\n=== Running KMeans (k={n_clusters}) ===")
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
cluster_labels_train = kmeans.fit_predict(X_train)
cluster_centers = kmeans.cluster_centers_


# ============================================================
# 5. 原型选择
# ============================================================
print("\n=== Selecting prototype molecules ===")
prototypes = []
for c in range(n_clusters):
    idx_c = np.where(cluster_labels_train == c)[0]
    if len(idx_c) == 0:
        prototypes.append(None)
        continue

    X_c = X_train[idx_c]
    center = cluster_centers[c]
    dist = np.linalg.norm(X_c - center, axis=1)
    proto_idx = idx_c[np.argmin(dist)]
    prototypes.append(proto_idx)


# ============================================================
# 6. cluster 统计输出
# ============================================================
cluster_rows = []
for c in range(n_clusters):
    idx_c = np.where(cluster_labels_train == c)[0]
    n_c = len(idx_c)

    if n_c > 0:
        mean_logN = float(np.mean(y_train_log[idx_c]))
        std_logN  = float(np.std(y_train_log[idx_c]))
        mean_N    = float(np.mean(y_train[idx_c]))
        std_N     = float(np.std(y_train[idx_c]))
    else:
        mean_logN = std_logN = mean_N = std_N = np.nan

    proto_idx = prototypes[c]
    if proto_idx is not None:
        proto_selfies = train_selfies[proto_idx]
        proto_name    = lookup_name_by_selfies(proto_selfies, selfies_to_name)
    else:
        proto_selfies = "NA"
        proto_name = "NA"

    cluster_rows.append({
        "cluster_id": c,
        "n_members": n_c,
        "mean_log10_N": mean_logN,
        "std_log10_N": std_logN,
        "mean_N": mean_N,
        "std_N": std_N,
        "prototype_train_index": proto_idx if proto_idx is not None else -1,
        "prototype_selfies": proto_selfies,
        "prototype_name": proto_name,
    })

cluster_df = pd.DataFrame(cluster_rows)
cluster_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_statistics.csv"), index=False)
print("Saved cluster_statistics.csv")


# ============================================================
# 7. prototype_table 输出
# ============================================================
proto_df = cluster_df[cluster_df["prototype_train_index"] >= 0][[
    "cluster_id", "prototype_train_index", "prototype_selfies",
    "prototype_name", "mean_log10_N", "mean_N", "n_members"
]]
proto_df.to_csv(os.path.join(OUTPUT_DIR, "prototype_table.csv"), index=False)
print("Saved prototype_table.csv")


# ============================================================
# 8. 测试集 → 原型映射
# ============================================================
print("\n=== Mapping test molecules to prototypes ===")
test_cluster_labels = kmeans.predict(X_test)

test_rows = []
for i in range(X_test.shape[0]):
    c = int(test_cluster_labels[i])
    proto_idx = prototypes[c]

    test_self = test_selfies[i]
    test_name = lookup_name_by_selfies(test_self, selfies_to_name)

    if proto_idx is not None:
        proto_self = train_selfies[proto_idx]
        proto_name = lookup_name_by_selfies(proto_self, selfies_to_name)
        stats = cluster_df.loc[cluster_df["cluster_id"] == c].iloc[0]
        mean_logN_c = stats["mean_log10_N"]
        mean_N_c = stats["mean_N"]
    else:
        proto_self = proto_name = "NA"
        mean_logN_c = mean_N_c = np.nan

    test_rows.append({
        "test_index": i,
        "test_selfies": test_self,
        "test_name": test_name,
        "assigned_cluster": c,
        "prototype_train_index": proto_idx if proto_idx is not None else -1,
        "prototype_selfies": proto_self,
        "prototype_name": proto_name,
        "cluster_mean_log10_N": mean_logN_c,
        "cluster_mean_N": mean_N_c,
    })

test_proto_df = pd.DataFrame(test_rows)
test_proto_df.to_csv(os.path.join(OUTPUT_DIR, "test_to_prototype.csv"), index=False)
print("Saved test_to_prototype.csv")


# ============================================================
# 9. t-SNE 可视化
# ============================================================
print("\n=== Running t-SNE (this may take 10–20 sec) ===")
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
X_tsne = tsne.fit_transform(X_train)

proto_indices = cluster_df["prototype_train_index"].values

print("Drawing t-SNE figure ...")
plt.figure(figsize=(10,8))

plt.scatter(
    X_tsne[:,0], X_tsne[:,1],
    c=cluster_labels_train,
    cmap="tab20",
    alpha=0.65,
    s=35
)

# 标注 prototype
for cid, proto_idx in enumerate(proto_indices):
    if proto_idx < 0:
        continue
    x, y = X_tsne[proto_idx]
    plt.scatter(x, y, c="black", s=160, marker="*", edgecolor="white", linewidth=1.3)
    plt.text(x+1.5, y+1.5, f"C{cid}", fontsize=9, weight="bold")

plt.title("t-SNE Visualization of Molecular Embeddings\nClusters + Prototype Molecules (stars)")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.tight_layout()

tsne_path = os.path.join(OUTPUT_DIR, "tsne_clusters.png")
plt.savefig(tsne_path, dpi=600)
plt.show()
print("Saved t-SNE figure to:", tsne_path)

print("\n=== Pipeline Finished Successfully ===\n")
