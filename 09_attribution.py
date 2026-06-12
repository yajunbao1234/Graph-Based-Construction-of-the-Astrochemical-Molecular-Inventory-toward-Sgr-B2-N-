######ig解释
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import seaborn as sns
from tqdm import tqdm
import pandas as pd

N_ITER = 15              # 👈 想跑几次就改这里
OUT_ROOT = "./data/test9"

for it in range(1, N_ITER + 1):

    print(f"\n===== Running iteration {it}/{N_ITER} =====")

    iter_tag = f"iter_{it:02d}"
    iter_dir = os.path.join(OUT_ROOT, iter_tag)
    os.makedirs(iter_dir, exist_ok=True)


    # 文件路径
    path_data = "./data/test7"
    path_label = "./data/test1"
    training_embedding_file = os.path.join(path_data, "training_data_embedding.txt")
    testing_embedding_file  = os.path.join(path_data, "testing_data_embedding.txt")
    training_label_file     = os.path.join(path_label, "training_label.txt")
    testing_label_file      = os.path.join(path_label, "testing_label.txt")

    # 日志文件
    log_file = "./data/test7/model_metrics_log.txt"
    log_f = open(log_file, "a", encoding="utf-8")

    log_f.write("\n============================\n")
    log_f.write("Experiment Time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log_f.write("============================\n")


    # -------------------------------
    # 1. 加载数据
    # -------------------------------
    X_train = np.loadtxt(training_embedding_file)
    X_test  = np.loadtxt(testing_embedding_file)

    # -------------------------------
    # 2. 读取 label 并转换科学计数法
    # -------------------------------
    def read_label(file_path):
        labels = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                line = line.replace("*10^", "e")
                labels.append(float(line))
        return np.array(labels)

    y_train = read_label(training_label_file)
    y_test  = read_label(testing_label_file)

    # -------------------------------
    # 3. log10 + 标准化
    # -------------------------------
    y_train_log = np.log10(y_train)
    y_test_log  = np.log10(y_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_log.reshape(-1,1)).ravel()
    y_test_scaled  = scaler_y.transform(y_test_log.reshape(-1,1)).ravel()

    # -------------------------------
    # 4. 随机森林 + 线性回归
    # -------------------------------
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features='sqrt',
        min_samples_leaf=2,
        bootstrap=True,
        random_state=None
    )
    rf.fit(X_train, y_train_scaled)

    y_pred_rf_scaled = rf.predict(X_test)
    y_pred_rf = scaler_y.inverse_transform(y_pred_rf_scaled.reshape(-1,1)).ravel()

    lr = LinearRegression()
    lr.fit(X_train, y_train_scaled)
    y_pred_lr_scaled = lr.predict(X_test)
    y_pred_lr = scaler_y.inverse_transform(y_pred_lr_scaled.reshape(-1,1)).ravel()

    # -------------------------------
    # 5. 评估模型
    # -------------------------------
    def evaluate(y_true_log, y_pred_log, model_name):
        mse = mean_squared_error(y_true_log, y_pred_log)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_log, y_pred_log)

        print(f"{model_name} performance on log scale:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}\n")

        log_f.write(f"{model_name} performance on log scale:\n")
        log_f.write(f"RMSE: {rmse:.4f}\n")
        log_f.write(f"R2:   {r2:.4f}\n\n")

    evaluate(y_test_log, y_pred_rf, "Random Forest")
    evaluate(y_test_log, y_pred_lr, "Linear Regression")


    # -------------------------------
    # 6. 可视化
    # -------------------------------
    plt.figure(figsize=(6,6))
    plt.scatter(y_test_log, y_pred_rf, alpha=0.6, label="RF Predicted")
    plt.scatter(y_test_log, y_pred_lr, alpha=0.6, label="LR Predicted", marker='x')
    plt.plot([min(y_test_log), max(y_test_log)],
             [min(y_test_log), max(y_test_log)],
             color='red', linestyle='--', label='Ideal')
    plt.xlabel("True log10(Density)")
    plt.ylabel("Predicted log10(Density)")
    plt.legend()
    plt.title("Predicted vs True log10(Density)")
    plt.show()


    # -------------------------------
    # 7. Integrated Gradients（加速版）
    # -------------------------------
    def numerical_ig(model, x, baseline=None, steps=20):   # ✔ steps=20（提速）
        x = x.reshape(1, -1)
        if baseline is None:
            baseline = np.zeros_like(x)

        eps = 1e-4
        grads = []

        for k in range(steps + 1):
            alpha = k / steps
            s = baseline + alpha * (x - baseline)

            grad_s = np.zeros_like(x)
            for i in range(x.shape[1]):
                s_eps = s.copy()
                s_eps[0, i] += eps
                grad_s[0, i] = (model.predict(s_eps)[0] - model.predict(s)[0]) / eps

            grads.append(grad_s)

        avg_grad = np.mean(np.vstack(grads), axis=0)
        ig = (x - baseline) * avg_grad
        return ig.flatten()

    # -------------------------------
    # 8. 六变量切片
    # -------------------------------
    total_dim = X_train.shape[1]
    dim_per_var = total_dim // 6
    print("每个变量维度 =", dim_per_var)

    var_slices = {
        "source_size":         slice(0*dim_per_var, 1*dim_per_var),
        "rotation_temp":     slice(1*dim_per_var, 2*dim_per_var),
        "line_width":          slice(2*dim_per_var, 3*dim_per_var),
        "velocity_offset":     slice(3*dim_per_var, 4*dim_per_var),
        "vibrational_state":   slice(4*dim_per_var, 5*dim_per_var),
        "selfies_structure":   slice(5*dim_per_var, 6*dim_per_var),
    }

    # -------------------------------
    # 9. 加速：只对 30 个样本做 IG
    # -------------------------------
    K = min(30, X_test.shape[0])   # ✔ 只算前30个（论文足够）

    dim_ig_list = []
    var_ig_list = []

    for i in tqdm(range(K)):
        x = X_test[i]
        ig = numerical_ig(rf, x)

        dim_ig_list.append(ig)

        var_scores = {
            "source_size":         ig[var_slices["source_size"]].sum(),
            "rotation_temp":     ig[var_slices["rotation_temp"]].sum(),
            "line_width":          ig[var_slices["line_width"]].sum(),
            "velocity_offset":     ig[var_slices["velocity_offset"]].sum(),
            "vibrational_state":   ig[var_slices["vibrational_state"]].sum(),
            "selfies_structure":   ig[var_slices["selfies_structure"]].sum(),
        }
        var_ig_list.append(var_scores)

    dim_ig_array = np.array(dim_ig_list)
    var_ig_df = pd.DataFrame(var_ig_list)

    np.savetxt(
        os.path.join(iter_dir, "IG_dimension_importance.txt"),
        dim_ig_array
    )
    var_ig_df.to_csv(
        os.path.join(iter_dir, "IG_variable_importance.csv"),
        index=False
    )

    print("IG 结果已保存。")

    # -------------------------------
    # 10. 六变量图
    # -------------------------------
    sns.set_theme(style="whitegrid", font_scale=1.3)

    # 先把列名换成论文友好的显示名（不改数据本身，只改画图显示）
    pretty_names = {
        "source_size": "Source size",
        "rotation_temp": "Rotation temp.",
        "line_width": "Line width",
        "velocity_offset": "Velocity offset",
        "vibrational_state": "Vibrational state",
        "selfies_structure": "SELFIES structure",
    }

    mean_vals = var_ig_df.mean().rename(index=pretty_names)
    std_vals = var_ig_df.std().rename(index=pretty_names)

    plt.figure(figsize=(8, 5))
    plt.bar(
        mean_vals.index, mean_vals.values,
        yerr=std_vals.values,
        capsize=5
    )


    plt.ylabel("Path-integrated attribution (IG, log$_{10}$ $N$)")

    # plt.ylabel("Integrated Gradients attribution")

    plt.xlabel("Input variable")  # ✅ 横轴标题补上
    plt.title("Variable-level attribution")

    plt.xticks(
        fontsize=10,
        rotation=20,
        ha="right"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(iter_dir, "fig_variable_importance.png"), dpi=600)

    # -------------------------------
    # 11. 维度 IG 热力图
    # -------------------------------
    avg_ig = dim_ig_array.mean(axis=0)

    plt.figure(figsize=(14, 2.5))
    ax = sns.heatmap(
        avg_ig[np.newaxis, :],
        cmap="viridis",
        cbar=True,
        xticklabels=False,  # 维度太多就不逐个显示刻度
        yticklabels=["Average"],
        cbar_kws={"shrink": 0.8}
    )

    # ✅ 补坐标轴标题
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("")  # y 轴只显示 "Average" 就够了，也可以写 "Aggregation"
    plt.title("Embedding dimension-level attribution")

    plt.tight_layout()
    plt.savefig(os.path.join(iter_dir, "fig_dimension_heatmap.png"), dpi=600)

    # -------------------------------
    # 12. 关闭日志文件
    # -------------------------------
    log_f.close()
    print(f"日志已追加保存到 {log_file}")
    print("IG 解释 + 图像生成全部完成！🎉")
