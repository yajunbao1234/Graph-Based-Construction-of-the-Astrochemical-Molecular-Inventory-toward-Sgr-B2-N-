import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 路径配置
# ============================================================
path_data = "./data/test7"
path_label = "./data/test1"

TRAIN_EMB_FILE  = os.path.join(path_data, "training_data_embedding.txt")
TEST_EMB_FILE   = os.path.join(path_data, "testing_data_embedding.txt")

TRAIN_LABEL_TRUE = os.path.join(path_label, "training_label.txt")
TEST_LABEL_TRUE  = os.path.join(path_label, "testing_label.txt")
TEST_LABEL_PRED  = os.path.join(path_data, "testing_label_pred.txt")

TRAIN_DATA_TXT  = os.path.join(path_label, "training_data.txt")
TEST_DATA_TXT   = os.path.join(path_label, "testing_data.txt")

MOLECULE_TABLE  = os.path.join(path_label, "Core_Components_reordered.txt")

OUTPUT_DIR = "./data/test12"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 工具函数
# ============================================================
def read_selfies_from_last_column(file_path):
    selfies_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                selfies_list.append(line.split()[-1])
    return selfies_list


def build_selfies_to_name_map(table_path):
    mapping = {}
    with open(table_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) > 9:
                mapping[parts[7]] = parts[9]
    return mapping


def lookup_name_by_selfies(selfies, s2n_map):
    return s2n_map.get(selfies, "UNKNOWN")


def read_label_linear(file_path):
    vals = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().replace("*10^", "e")
            if s:
                vals.append(float(s))
    return np.array(vals)


def is_common_CO_tracer(name):               # ★ NEW
    """
    判断是否为常规强丰度 CO 类 tracer
    """
    if name is None:
        return False
    name = name.upper()
    return (
        name == "CO" or
        "13CO" in name or
        "C18O" in name or
        "C17O" in name
    )


# ============================================================
# 3. 读取数据
# ============================================================
X_train = np.loadtxt(TRAIN_EMB_FILE)
X_test  = np.loadtxt(TEST_EMB_FILE)

N_pred_test = read_label_linear(TEST_LABEL_PRED)
logN_pred_test = np.log10(N_pred_test)

train_selfies = read_selfies_from_last_column(TRAIN_DATA_TXT)
test_selfies  = read_selfies_from_last_column(TEST_DATA_TXT)

selfies_to_name = build_selfies_to_name_map(MOLECULE_TABLE)


# ============================================================
# 4. 局部相似度密度
# ============================================================
nbrs = NearestNeighbors(n_neighbors=10, metric="cosine")
nbrs.fit(X_train)

distances, _ = nbrs.kneighbors(X_test)
local_density = (1.0 - distances).mean(axis=1)


# ============================================================
# 5. 结构新颖性
# ============================================================
kmeans = KMeans(n_clusters=8, random_state=0, n_init=10)
kmeans.fit(X_train)

test_clusters = kmeans.predict(X_test)
centers = kmeans.cluster_centers_

novelty = np.linalg.norm(
    X_test - centers[test_clusters],
    axis=1
)


# ============================================================
# 5.5 相对 CO 子空间的新颖性
# ============================================================
co_indices = []
for i, sf in enumerate(test_selfies):
    name = lookup_name_by_selfies(sf, selfies_to_name)
    if is_common_CO_tracer(name):
        co_indices.append(i)

if len(co_indices) > 0:
    co_center = X_test[co_indices].mean(axis=0)
else:
    co_center = X_train.mean(axis=0)

relative_novelty = np.linalg.norm(X_test - co_center, axis=1)


# ============================================================
# 6. 综合优先级得分
# ============================================================
features = np.vstack([
    logN_pred_test,
    local_density,
    novelty,
    relative_novelty          # ★ NEW
]).T

features_scaled = StandardScaler().fit_transform(features)

logN_s, dens_s, nov_s, relnov_s = features_scaled.T

relnov_s = np.tanh(relnov_s)


w1, w2, w3, w4 = 0.1, 0.2, 0.2, 0.5

final_score = (
    w1 * logN_s +
    w2 * dens_s +
    w3 * nov_s +
    w4 * relnov_s
)


# ============================================================
# 7. 排序 CSV
# ============================================================
rows = []
for i in range(len(X_test)):
    sf = test_selfies[i]
    name = lookup_name_by_selfies(sf, selfies_to_name)
    rows.append({
        "test_index": i,
        "name": name,
        "selfies": sf,
        "log10_N_pred": logN_pred_test[i],
        "local_density": local_density[i],
        "novelty": novelty[i],
        "relative_novelty": relative_novelty[i],   # ★ NEW
        "final_score": final_score[i]
    })

priority_df = pd.DataFrame(rows)
priority_df = priority_df.sort_values("final_score", ascending=False).reset_index(drop=True)
priority_df["rank"] = priority_df.index + 1

csv_path = os.path.join(OUTPUT_DIR, "priority_ranking.csv")
priority_df.to_csv(csv_path, index=False)


