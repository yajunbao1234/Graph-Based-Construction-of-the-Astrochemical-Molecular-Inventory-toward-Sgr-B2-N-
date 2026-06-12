import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


# ====================================================
# 1. 路径配置
# ====================================================
path_data = "./data/test7"
path_label = "./data/test1"

TRAIN_EMB_FILE      = os.path.join(path_data, "training_data_embedding.txt")
TRAIN_LABEL_FILE    = os.path.join(path_label, "training_label.txt")

TRAIN_DATA_TXT      = os.path.join(path_label, "training_data.txt")
MOLECULE_TABLE_FILE = os.path.join(path_label, "Core_Components_reordered.txt")

OUTPUT_DIR = "./data/test11"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ====================================================
# 2. 工具函数
# ====================================================
def read_label(file_path):
    vals = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip().replace("*10^", "e")
            vals.append(float(line))
    return np.array(vals)


def read_selfies_last_column(txt_file):
    out = []
    with open(txt_file, "r") as f:
        for line in f:
            if line.strip():
                parts = line.split()
                out.append(parts[-1])
    return out


def load_selfies_to_name(table_file):
    mapping = {}
    with open(table_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 9:
                selfies = parts[7]
                name = parts[9]
                mapping[selfies] = name
    return mapping


# ====================================================
# 3. 加载数据
# ====================================================
print("Loading data ...")

X = np.loadtxt(TRAIN_EMB_FILE)
y = read_label(TRAIN_LABEL_FILE)
y_log = np.log10(y)

train_selfies = read_selfies_last_column(TRAIN_DATA_TXT)
selfies_to_name = load_selfies_to_name(MOLECULE_TABLE_FILE)

N = X.shape[0]
print(f"Loaded {N} training samples.")


# ====================================================
# 4. 计算 embedding 距离矩阵
# ====================================================
print("Computing pairwise embedding distances ...")
dist_matrix = pairwise_distances(X, metric="euclidean")


# ====================================================
# 5. 计算丰度差异矩阵
# ====================================================
print("Computing abundance difference matrix ...")
diff_matrix = np.abs(y_log[:, None] - y_log[None, :])


# ====================================================
# 6. 自动阈值（可调）
# ====================================================
dist_thr = np.percentile(dist_matrix, 15)   # 前 15% 最近邻
diff_thr = np.percentile(diff_matrix, 85)   # 后 15% 最大差异

print(f"Distance threshold = {dist_thr:.4f}")
print(f"Abundance difference threshold = {diff_thr:.4f}")


# ====================================================
# 7. 提取对比样本对
# ====================================================
rows = []

for i in range(N):
    for j in range(i + 1, N):  # 避免重复 pair
        d = dist_matrix[i, j]
        df = diff_matrix[i, j]

        if d < dist_thr and df > diff_thr:
            si = train_selfies[i]
            sj = train_selfies[j]

            name_i = selfies_to_name.get(si, "UNKNOWN")
            name_j = selfies_to_name.get(sj, "UNKNOWN")

            rows.append({
                "i_index": i,
                "j_index": j,
                "i_selfies": si,
                "j_selfies": sj,
                "i_name": name_i,
                "j_name": name_j,
                "dist": d,
                "log10_N_i": y_log[i],
                "log10_N_j": y_log[j],
                "diff": df
            })

contrast_df = pd.DataFrame(rows)
contrast_df.to_csv(os.path.join(OUTPUT_DIR, "contrastive_pairs.csv"), index=False)

print(f"Found {len(contrast_df)} contrastive pairs.")
print("Saved contrastive_pairs.csv")


# ====================================================
# 8. 可视化：结构距离 vs 丰度差异
# ====================================================
plt.figure(figsize=(8, 6))

plt.scatter(contrast_df["dist"], contrast_df["diff"], s=40, alpha=0.7, c='red')
plt.xlabel("Embedding Distance")
plt.ylabel("Abundance Difference |Δ log10(N)|")
plt.title("Contrastive Pairs: Similar Structure but Large Abundance Difference")
plt.grid(True)

out_fig = os.path.join(OUTPUT_DIR, "contrastive_pairs_plot.png")
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.show()

print("Saved plot:", out_fig)


# ====================================================
# 9. 分子对 + embedding similarity score + column densities
# ====================================================
print("Generating reviewer-style pair figure ...")

if len(contrast_df) > 0:
    pair_df = contrast_df.copy()

    # 将距离转成相似度分数（距离越小，相似度越高）
    pair_df["similarity_score"] = 1.0 / (1.0 + pair_df["dist"])

    # 选取最有代表性的若干组：优先距离更小、丰度差更大
    top_n = min(12, len(pair_df))
    pair_df = pair_df.sort_values(
        by=["dist", "diff"],
        ascending=[True, False]
    ).head(top_n).copy()

    # 若名称缺失，则退回 SELFIES
    def display_name(name, selfies):
        if isinstance(name, str) and name != "UNKNOWN":
            return name
        return selfies

    pair_df["i_display"] = [
        display_name(n, s) for n, s in zip(pair_df["i_name"], pair_df["i_selfies"])
    ]
    pair_df["j_display"] = [
        display_name(n, s) for n, s in zip(pair_df["j_name"], pair_df["j_selfies"])
    ]
    pair_df["pair_label"] = pair_df["i_display"] + "  vs  " + pair_df["j_display"]

    # 反转顺序，保证最优 pair 显示在图顶端
    pair_df = pair_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(pair_df))

    fig_h = max(6, 0.65 * len(pair_df) + 2)
    plt.figure(figsize=(13, fig_h))

    # 连接同一对分子的两端 column density
    for k, row in pair_df.iterrows():
        x1 = row["log10_N_i"]
        x2 = row["log10_N_j"]
        plt.plot([x1, x2], [k, k], linewidth=2, alpha=0.8)

    # 两端点：分别表示 pair 中的两个分子
    plt.scatter(pair_df["log10_N_i"], y_pos, s=90, marker="o", label="Molecule i")
    plt.scatter(pair_df["log10_N_j"], y_pos, s=90, marker="s", label="Molecule j")

    plt.yticks(y_pos, pair_df["pair_label"])
    plt.xlabel("log10(Column Density)")
    plt.ylabel("Representative Molecular Pairs")
    plt.title("Representative Contrastive Molecular Pairs\nEmbedding Similarity Scores and Column Densities")

    # 给每一组 pair 添加 similarity score 和 distance 注释
    x_min = min(pair_df["log10_N_i"].min(), pair_df["log10_N_j"].min())
    x_max = max(pair_df["log10_N_i"].max(), pair_df["log10_N_j"].max())
    x_span = x_max - x_min
    if x_span == 0:
        x_span = 1.0
    text_pad = 0.04 * x_span

    for k, row in pair_df.iterrows():
        x_right = max(row["log10_N_i"], row["log10_N_j"])
        txt = f"sim={row['similarity_score']:.3f}, d={row['dist']:.3f}"
        plt.text(x_right + text_pad, k, txt, va="center", fontsize=9)

    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_fig2 = os.path.join(OUTPUT_DIR, "contrastive_pairs_pairs_density.png")
    plt.savefig(out_fig2, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved reviewer-style pair figure:", out_fig2)

else:
    print("No contrastive pairs found. Reviewer-style pair figure was not generated.")

print("\n==== Step 5 Completed ====\n")