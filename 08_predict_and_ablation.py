import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

if len(sys.argv) < 4:
    raise ValueError("请从 main.py 传入 iteration、seed、data_tag，例如：python test8.py 1 6612345 A01")

iteration = int(sys.argv[1])
seed = int(sys.argv[2])
data_tag = sys.argv[3]

np.random.seed(seed)
random.seed(seed)


# -------------------------------
# 文件路径
# -------------------------------
path_data = "./data/test7"
path_label = "./data/test1"

# 按 A01/A02/... 分开保存
path_output_root = os.path.join("./data/test8", data_tag)
os.makedirs(path_output_root, exist_ok=True)

iter_folder_name = f"iter_{iteration:03d}_seed_{seed}"
path_output = os.path.join(path_output_root, iter_folder_name)
os.makedirs(path_output, exist_ok=True)

# 分成 prediction / ablation 两个子目录
pred_dir = os.path.join(path_output, "prediction")
ablation_dir = os.path.join(path_output, "ablation")
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(ablation_dir, exist_ok=True)

training_embedding_file = os.path.join(path_data, "training_data_embedding.txt")
testing_embedding_file = os.path.join(path_data, "testing_data_embedding.txt")
training_label_file = os.path.join(path_label, "training_label.txt")
testing_label_file = os.path.join(path_label, "testing_label.txt")
testing_data_file = os.path.join(path_label, "testing_data.txt")

metrics_all = []


# -------------------------------
# 1. embedding 结构
# -------------------------------
EMB_DIM = 32   # 如果不是 32，请改这里

IDX = {
    "source_size": (0 * EMB_DIM, 1 * EMB_DIM),
    "temperature": (1 * EMB_DIM, 2 * EMB_DIM),
    "linewidth":   (2 * EMB_DIM, 3 * EMB_DIM),
    "velocity":    (3 * EMB_DIM, 4 * EMB_DIM),
    "vibration":   (4 * EMB_DIM, 5 * EMB_DIM),
    "selfies":     (5 * EMB_DIM, 6 * EMB_DIM),
}


def select_embedding_blocks(X, keep_blocks):
    cols = []
    for name in keep_blocks:
        s, e = IDX[name]
        cols.extend(range(s, e))
    return X[:, cols]


# -------------------------------
# 2. 加载 embedding
# -------------------------------
X_train_full = np.loadtxt(training_embedding_file)
X_test_full = np.loadtxt(testing_embedding_file)

# 主预测默认使用全部 embedding
X_train = X_train_full.copy()
X_test = X_test_full.copy()


# -------------------------------
# 3. 读取 label
# -------------------------------
def read_label_with_raw(file_path):
    raw_labels = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw:
                raw_labels.append(raw)
                labels.append(float(raw.replace("*10^", "e")))
    return raw_labels, np.array(labels, dtype=float)


train_label_raw, y_train = read_label_with_raw(training_label_file)
test_label_raw, y_test = read_label_with_raw(testing_label_file)


# -------------------------------
# 4. 读取 testing_data.txt 原始参数
# -------------------------------
def read_testing_data_with_raw(file_path):
    raw_lines = []
    split_rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if raw.strip():
                raw_lines.append(raw)
                split_rows.append(raw.strip().split())

    return raw_lines, split_rows


testing_data_raw_lines, testing_data_split_rows = read_testing_data_with_raw(testing_data_file)


# -------------------------------
# 5. 转换为 log10(N) 并标准化
# -------------------------------
y_train_log = np.log10(y_train)
y_test_log = np.log10(y_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_log.reshape(-1, 1)).ravel()


# -------------------------------
# 6. 主预测评估函数
# -------------------------------
def evaluate_main(y_true_log, y_pred_log, model_name):
    mse = mean_squared_error(y_true_log, y_pred_log)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_log, y_pred_log)
    r2 = r2_score(y_true_log, y_pred_log)
    spearman_rho, _ = spearmanr(y_true_log, y_pred_log)

    return {
        "data_tag": data_tag,
        "iteration": iteration,
        "seed": seed,
        "Model": model_name,
        "MSE": mse,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Spearman_rho": spearman_rho
    }


# -------------------------------
# 7. 消融评估函数
# -------------------------------
def evaluate_ablation(y_true_log, y_pred_log, setting_name):
    mse = mean_squared_error(y_true_log, y_pred_log)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_log, y_pred_log)
    r2 = r2_score(y_true_log, y_pred_log)
    spearman_rho, _ = spearmanr(y_true_log, y_pred_log)

    return {
        "data_tag": data_tag,
        "iteration": iteration,
        "seed": seed,
        "Setting": setting_name,
        "MSE": mse,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Spearman_rho": spearman_rho
    }


# -------------------------------
# 8. 主预测：RF / LR / GBR / GPR
# -------------------------------
rf = RandomForestRegressor(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    bootstrap=True,
    random_state=seed
)
rf.fit(X_train, y_train_scaled)
y_pred_rf_scaled = rf.predict(X_test)
y_pred_rf = scaler_y.inverse_transform(y_pred_rf_scaled.reshape(-1, 1)).ravel()
metrics_all.append(evaluate_main(y_test_log, y_pred_rf, "Random Forest"))

lr = LinearRegression()
lr.fit(X_train, y_train_scaled)
y_pred_lr_scaled = lr.predict(X_test)
y_pred_lr = scaler_y.inverse_transform(y_pred_lr_scaled.reshape(-1, 1)).ravel()
metrics_all.append(evaluate_main(y_test_log, y_pred_lr, "Linear Regression"))

gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=seed
)
gbr.fit(X_train, y_train_scaled)
y_pred_gbr_scaled = gbr.predict(X_test)
y_pred_gbr = scaler_y.inverse_transform(y_pred_gbr_scaled.reshape(-1, 1)).ravel()
metrics_all.append(evaluate_main(y_test_log, y_pred_gbr, "Gradient Boosting"))

kernel = C(1.0, (1e-3, 1e3)) * RBF(
    length_scale=1.0,
    length_scale_bounds=(1e-2, 1e2)
) + WhiteKernel(
    noise_level=1e-3,
    noise_level_bounds=(1e-6, 1e0)
)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=3,
    normalize_y=False,
    random_state=seed
)

try:
    gpr.fit(X_train, y_train_scaled)
    y_pred_gpr_scaled = gpr.predict(X_test)
    y_pred_gpr = scaler_y.inverse_transform(y_pred_gpr_scaled.reshape(-1, 1)).ravel()
    metrics_all.append(evaluate_main(y_test_log, y_pred_gpr, "Gaussian Process"))
    gpr_success = True
except Exception as e:
    print("Gaussian Process Regression failed:", str(e))
    y_pred_gpr = None
    gpr_success = False


# -------------------------------
# 9. 保存主预测指标
# -------------------------------
metrics_df = pd.DataFrame(metrics_all)
metrics_save_file = os.path.join(pred_dir, "metrics_summary.csv")
metrics_df.to_csv(metrics_save_file, index=False, encoding="utf-8-sig")


# -------------------------------
# 10. 长度对齐
# -------------------------------
n_samples = min(
    len(y_test_log),
    len(y_test),
    len(test_label_raw),
    len(testing_data_raw_lines),
    len(testing_data_split_rows),
    len(y_pred_rf),
    len(y_pred_lr),
    len(y_pred_gbr)
)
if gpr_success:
    n_samples = min(n_samples, len(y_pred_gpr))

if n_samples == 0:
    raise ValueError("测试集为空，无法保存结果")

y_test_log_aligned = y_test_log[:n_samples]
y_test_aligned = y_test[:n_samples]
test_label_raw_aligned = test_label_raw[:n_samples]
testing_data_raw_lines_aligned = testing_data_raw_lines[:n_samples]
testing_data_split_rows_aligned = testing_data_split_rows[:n_samples]
y_pred_rf_aligned = y_pred_rf[:n_samples]
y_pred_lr_aligned = y_pred_lr[:n_samples]
y_pred_gbr_aligned = y_pred_gbr[:n_samples]
if gpr_success:
    y_pred_gpr_aligned = y_pred_gpr[:n_samples]


# -------------------------------
# 11. 自动拆分原始参数列
# -------------------------------
max_cols = max(len(row) for row in testing_data_split_rows_aligned)

input_cols_dict = {}
for col_idx in range(max_cols):
    input_cols_dict[f"input_col_{col_idx+1}"] = [
        row[col_idx] if col_idx < len(row) else "" for row in testing_data_split_rows_aligned
    ]


# -------------------------------
# 12. 保存主预测 CSV 明细
# -------------------------------
pred_detail = pd.DataFrame({
    "data_tag": [data_tag] * n_samples,
    "iteration": [iteration] * n_samples,
    "seed": [seed] * n_samples,
    "sample_id": np.arange(1, n_samples + 1),
    "testing_data_raw": testing_data_raw_lines_aligned,
    "testing_label_raw": test_label_raw_aligned,
    "y_true_N": y_test_aligned,
    "y_true_log10N": y_test_log_aligned,
    "rf_pred_log10N": y_pred_rf_aligned,
    "rf_pred_N": 10 ** y_pred_rf_aligned,
    "lr_pred_log10N": y_pred_lr_aligned,
    "lr_pred_N": 10 ** y_pred_lr_aligned,
    "gbr_pred_log10N": y_pred_gbr_aligned,
    "gbr_pred_N": 10 ** y_pred_gbr_aligned
})

insert_pos = 5
for col_name, col_values in input_cols_dict.items():
    pred_detail.insert(insert_pos, col_name, col_values)
    insert_pos += 1

if gpr_success:
    pred_detail["gpr_pred_log10N"] = y_pred_gpr_aligned
    pred_detail["gpr_pred_N"] = 10 ** y_pred_gpr_aligned

pred_detail_file = os.path.join(pred_dir, "prediction_detail.csv")
pred_detail.to_csv(pred_detail_file, index=False, encoding="utf-8-sig")


# -------------------------------
# 13. 保存主预测 TXT 明细
# -------------------------------
txt_file = os.path.join(pred_dir, "prediction_detail.txt")

header_cols = [
    "data_tag",
    "sample_id",
    "testing_data_raw",
    "testing_label_raw",
    "y_true_N",
    "rf_pred_N",
    "lr_pred_N",
    "gbr_pred_N"
]
if gpr_success:
    header_cols.append("gpr_pred_N")

with open(txt_file, "w", encoding="utf-8") as f:
    f.write("\t".join(header_cols) + "\n")

    for i in range(n_samples):
        row = [
            data_tag,
            str(i + 1),
            testing_data_raw_lines_aligned[i],
            test_label_raw_aligned[i],
            f"{y_test_aligned[i]:.6e}",
            f"{(10 ** y_pred_rf_aligned[i]):.6e}",
            f"{(10 ** y_pred_lr_aligned[i]):.6e}",
            f"{(10 ** y_pred_gbr_aligned[i]):.6e}",
        ]
        if gpr_success:
            row.append(f"{(10 ** y_pred_gpr_aligned[i]):.6e}")

        f.write("\t".join(row) + "\n")


# -------------------------------
# 14. 主预测可视化
# -------------------------------
plt.figure(figsize=(6, 6))

plt.scatter(y_test_log_aligned, y_pred_rf_aligned, alpha=0.6, label="RF Predicted")
plt.scatter(y_test_log_aligned, y_pred_lr_aligned, alpha=0.6, label="LR Predicted", marker="x")
plt.scatter(y_test_log_aligned, y_pred_gbr_aligned, alpha=0.6, label="GBR Predicted", marker="^")

all_pred = [y_pred_rf_aligned, y_pred_lr_aligned, y_pred_gbr_aligned]

if gpr_success:
    plt.scatter(y_test_log_aligned, y_pred_gpr_aligned, alpha=0.6, label="GPR Predicted", marker="s")
    all_pred.append(y_pred_gpr_aligned)

vmin = min([np.min(y_test_log_aligned)] + [np.min(arr) for arr in all_pred])
vmax = max([np.max(y_test_log_aligned)] + [np.max(arr) for arr in all_pred])

plt.plot([vmin, vmax], [vmin, vmax], color="red", linestyle="--", label="Ideal")

mse_text_lines = []
for _, row in metrics_df.iterrows():
    mse_text_lines.append(f"{row['Model']}: MSE={row['MSE']:.4f}")

mse_text = "\n".join(mse_text_lines)
plt.text(
    0.03, 0.97,
    mse_text,
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

plt.xlabel("True log10(N)")
plt.ylabel("Predicted log10(N)")
plt.legend()
plt.title(f"{data_tag} | Predicted vs True log10(N), iter={iteration}, seed={seed}")

plot_file = os.path.join(pred_dir, f"{data_tag}_pred_vs_true_iter_{iteration:03d}_seed_{seed}.png")
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close()


# -------------------------------
# 15. 保留 RF 预测 txt
# -------------------------------
testing_label_pred_rf = 10 ** y_pred_rf_aligned
save_pred_file_rf = os.path.join(pred_dir, "testing_label_pred_rf.txt")
np.savetxt(save_pred_file_rf, testing_label_pred_rf, fmt="%.6e")


# -------------------------------
# 16. 消融实验设置
# -------------------------------
ablation_settings = {
    "Full model": [
        "source_size", "temperature", "linewidth",
        "velocity", "vibration", "selfies"
    ],
    "Physical-only": [
        "source_size", "temperature", "linewidth",
        "velocity", "vibration"
    ],
    "SELFIES-only": [
        "selfies"
    ],
    "No source size": [
        "temperature", "linewidth", "velocity",
        "vibration", "selfies"
    ],
    "No vibrational state": [
        "source_size", "temperature", "linewidth",
        "velocity", "selfies"
    ],
}


# -------------------------------
# 17. 运行消融实验
# -------------------------------
ablation_results = []
ablation_pred_dict = {
    "data_tag": [data_tag] * n_samples,
    "iteration": [iteration] * n_samples,
    "seed": [seed] * n_samples,
    "sample_id": np.arange(1, n_samples + 1),
    "testing_data_raw": testing_data_raw_lines_aligned,
    "testing_label_raw": test_label_raw_aligned,
    "y_true_N": y_test_aligned,
    "y_true_log10N": y_test_log_aligned,
}

for setting_name, blocks in ablation_settings.items():
    X_train_ab = select_embedding_blocks(X_train_full, blocks)
    X_test_ab = select_embedding_blocks(X_test_full, blocks)

    rf_ab = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        bootstrap=True,
        random_state=seed
    )
    rf_ab.fit(X_train_ab, y_train_scaled)

    y_pred_ab_scaled = rf_ab.predict(X_test_ab)
    y_pred_ab_log = scaler_y.inverse_transform(y_pred_ab_scaled.reshape(-1, 1)).ravel()
    y_pred_ab_log = y_pred_ab_log[:n_samples]

    ablation_results.append(
        evaluate_ablation(y_test_log_aligned, y_pred_ab_log, setting_name)
    )

    safe_name = setting_name.lower().replace(" ", "_").replace("-", "_")
    ablation_pred_dict[f"{safe_name}_pred_log10N"] = y_pred_ab_log
    ablation_pred_dict[f"{safe_name}_pred_N"] = 10 ** y_pred_ab_log


# -------------------------------
# 18. 保存消融实验汇总
# -------------------------------
ablation_df = pd.DataFrame(ablation_results)

ablation_csv = os.path.join(ablation_dir, "ablation_summary.csv")
ablation_json = os.path.join(ablation_dir, "ablation_summary.json")

ablation_df.to_csv(ablation_csv, index=False, encoding="utf-8-sig")
with open(ablation_json, "w", encoding="utf-8") as f:
    json.dump(ablation_results, f, indent=2, ensure_ascii=False)


# -------------------------------
# 19. 保存消融逐样本明细
# -------------------------------
ablation_pred_detail = pd.DataFrame(ablation_pred_dict)

insert_pos = 5
for col_name, col_values in input_cols_dict.items():
    ablation_pred_detail.insert(insert_pos, col_name, col_values)
    insert_pos += 1

ablation_pred_detail_file = os.path.join(ablation_dir, "ablation_prediction_detail.csv")
ablation_pred_detail.to_csv(ablation_pred_detail_file, index=False, encoding="utf-8-sig")


# -------------------------------
# 20. 单独保存 Full model 的预测值
# -------------------------------
full_model_pred_file = os.path.join(ablation_dir, "full_model_testing_label_pred.txt")
np.savetxt(
    full_model_pred_file,
    ablation_pred_detail["full_model_pred_N"].values,
    fmt="%.6e"
)


# -------------------------------
# 21. 控制台输出
# -------------------------------
print("=" * 60)
print(f"数据标签: {data_tag}")
print(f"迭代轮次: {iteration}")
print(f"随机种子: {seed}")
print(f"本次结果已保存到: {path_output}")
print("-" * 60)
print("[主预测]")
print(f"TXT 明细文件: {txt_file}")
print(f"指标文件: {metrics_save_file}")
print(f"图像文件: {plot_file}")
print(f"RF预测值文件: {save_pred_file_rf}")
print("-" * 60)
print("[消融实验]")
print(f"消融汇总 CSV: {ablation_csv}")
print(f"消融汇总 JSON: {ablation_json}")
print(f"消融逐样本明细: {ablation_pred_detail_file}")
print(f"Full model 预测值: {full_model_pred_file}")
print("=" * 60)
