import os

# ============================================
# ★★★ 可修改配置 ★★★
# ============================================
NUM_VARS = 6
path_in = "./data/test1"
path_out = "./data/test4"
total_data_file = os.path.join(path_in, "total_data.txt")


all_rows = []
with open(total_data_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()

        # 每个变量 = 两项（编号, 原始值）
        # 总长度 = NUM_VARS * 2
        expected_len = NUM_VARS * 2

        if len(parts) < expected_len:
            raise ValueError("行长度不足，NUM_VARS 设置可能不正确")

        ids = parts[0:expected_len:2]  # 编号   0,2,4,...
        values = parts[1:expected_len:2]  # 值     1,3,5,...

        # SELFIES（剩余全部 join）
        if len(parts) > expected_len:
            values[-1] = " ".join(parts[expected_len:])

        all_rows.append((ids, values))


single_var_files = []
for i in range(NUM_VARS):
    fp = open(os.path.join(path_out, f"v{i + 1}.txt"), "w", encoding="utf-8")
    single_var_files.append(fp)

# 两两组合文件 v1_v2.txt ... v(N-1)_vN.txt
pair_var_files = {}
for i in range(NUM_VARS):
    for j in range(i + 1, NUM_VARS):
        fname = f"v{i + 1}_v{j + 1}.txt"
        fp = open(os.path.join(path_out, fname), "w", encoding="utf-8")
        pair_var_files[(i, j)] = fp

# ============================================
# 写入数据
# ============================================

for ids, values in all_rows:

    # 单变量 (id, value)
    for i in range(NUM_VARS):
        single_var_files[i].write(ids[i] + "\t" + values[i] + "\n")

    # 两两组合 (只写 id)
    for i in range(NUM_VARS):
        for j in range(i + 1, NUM_VARS):
            pair_var_files[(i, j)].write(ids[i] + "\t" + ids[j] + "\n")

# ============================================
# 关闭文件
# ============================================
for f in single_var_files:
    f.close()
for f in pair_var_files.values():
    f.close()

print("test4 finished")
