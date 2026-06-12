import pandas as pd
import numpy as np
import re
import os
import sys
import random
from datetime import datetime

A_name = "A01"

input_file = f"./data/test1/areas_split/{A_name}.txt"

print(f"📂 当前处理文件: {input_file}")

train_data_file        = "./data/test1/training_data.txt"
train_data_total_file  = "./data/test1/training_data_total.txt"
test_data_file         = "./data/test1/testing_data.txt"
test_data_total_file   = "./data/test1/testing_data_total.txt"
train_label_file       = "./data/test1/training_label.txt"
test_label_file        = "./data/test1/testing_label.txt"

# test1 独立 seed 日志
seed_log_file          = "./data/test1/test1_used_seeds.csv"
seed_last_file         = "./data/test1/test1_last_seed.txt"


TRAIN_COLS = [1, 2, 3, 4, 5, 7]

# total 输出想保留的列（不展开 SELFIES）
TOTAL_COLS = [0, 1, 2, 3, 4, 5, 7]

# SELFIES 在 TRAIN_COLS 中的位置（自动算，不用人工改）
SELFIES_COL = TRAIN_COLS.index(7)


# ============================================================
# 读取数据
# ============================================================
df = pd.read_csv(input_file, sep="	", header=None)

test1_seed = random.SystemRandom().randint(1, 10**9)

df = df.sample(frac=1, random_state=test1_seed).reset_index(drop=True)
train_size = int(0.8 * len(df))

train_df = df.iloc[:train_size].reset_index(drop=True)
test_df  = df.iloc[train_size:].reset_index(drop=True)

# 记录 seed 到 ./data/test1
seed_record = pd.DataFrame([{
    "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "A_name": A_name,
    "test1_seed": test1_seed,
    "total_samples": len(df),
    "train_size": len(train_df),
    "test_size": len(test_df)
}])

if os.path.exists(seed_log_file):
    seed_record.to_csv(seed_log_file, mode="a", index=False, header=False, encoding="utf-8-sig")
else:
    seed_record.to_csv(seed_log_file, index=False, encoding="utf-8-sig")

with open(seed_last_file, "w", encoding="utf-8") as f:
    f.write(f"A_name={A_name}\n")
    f.write(f"test1_seed={test1_seed}\n")
    f.write(f"total_samples={len(df)}\n")
    f.write(f"train_size={len(train_df)}\n")
    f.write(f"test_size={len(test_df)}\n")
# ============================================================
# 1) total 输出（原样，不展开 SELFIES）
# ============================================================
train_total = train_df.iloc[:, TOTAL_COLS]
test_total  = test_df.iloc[:, TOTAL_COLS]

train_total.to_csv(train_data_total_file, sep="	", index=False, header=False)
test_total.to_csv(test_data_total_file,  sep="	", index=False, header=False)


# ============================================================
# 2) 模型训练的输入：TRAIN_COLS（展开 SELFIES）
# ============================================================
def select_and_reset(df, cols):
    df_new = df.iloc[:, cols].copy()
    df_new.columns = range(len(cols))  # 重新编号 0,1,2,...
    return df_new

train_data_raw = select_and_reset(train_df, TRAIN_COLS)
test_data_raw  = select_and_reset(test_df,  TRAIN_COLS)

# SELFIES 列自动计算后 = SELFIES_COL
def expand_selfies(df, col):
    df = df.copy()
    df[col] = df[col].apply(lambda x:
        "".join(re.findall(r'\[[^\]]+\]', str(x)))
    )
    return df

train_data = expand_selfies(train_data_raw, SELFIES_COL)
test_data  = expand_selfies(test_data_raw,  SELFIES_COL)

# 保存展开后的训练数据
train_data.to_csv(train_data_file, sep="	", index=False, header=False)
test_data.to_csv(test_data_file,  sep="	", index=False, header=False)


# ============================================================
# 3) 保存标签
# ============================================================
train_label = train_df.iloc[:, 8]
test_label  = test_df.iloc[:, 8]

train_label.to_csv(train_label_file, sep="	", index=False, header=False)
test_label.to_csv(test_label_file, sep="	", index=False, header=False)


# ============================================================
# Done
# ============================================================
print("🚀 完成全部 6 份文件生成！")
print(f"test1 独立 seed: {test1_seed}")
print(f"test1 seed 日志: {seed_log_file}")
print(f"test1 最近一次 seed 文件: {seed_last_file}")
print(f"训练集 total: {train_data_total_file}")
print(f"训练集用于 embedding: {train_data_file}")
print(f"训练集标签: {train_label_file}")
print(f"测试集 total: {test_data_total_file}")
print(f"测试集用于 embedding: {test_data_file}")
print(f"测试集标签: {test_label_file}")
print("test1 finished")
