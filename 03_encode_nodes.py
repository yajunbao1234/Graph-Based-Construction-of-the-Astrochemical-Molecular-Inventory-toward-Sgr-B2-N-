import os

# ====================== 可修改区域（最上面） ======================
NUM_FIELDS = 6
SELFIES_START_COL = 5
FIELD_OFFSETS_BASE = 1000000

# 输入输出路径
path_in = './data/test2'
path_out = './data/test3'
path_out_in = './data/test1'
input_file = os.path.join(path_in, "training_testing_combined.txt")

# ====================== 自动构建编号前缀 ======================
FIELD_OFFSETS = {
    i: FIELD_OFFSETS_BASE * (i + 1) for i in range(NUM_FIELDS)
}
# 例如：4列 → {0:1000000, 1:2000000, 2:3000000, 3:4000000}

# ====================== 保存所有字段编号映射 ======================
index_maps = {i: {} for i in range(NUM_FIELDS)}

def encode_value(col_idx, value):
    mapping = index_maps[col_idx]
    if value not in mapping:
        mapping[value] = FIELD_OFFSETS[col_idx] + len(mapping)
    return str(mapping[value])

# ====================== 主处理逻辑 ======================
molecule_data_index = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()

        # -------- SELFIES 合并 --------
        # 只合并从 SELFIES_START_COL 开始的所有内容
        selfies = " ".join(parts[SELFIES_START_COL:])

        # 替换 parts（前 SELFIES_START_COL 项 + 合并后 SELFIES）
        parts = parts[:SELFIES_START_COL] + [selfies]

        # 编码行
        encoded_row = []
        for col_idx, val in enumerate(parts):
            encoded_row.append(encode_value(col_idx, val))  # 编号
            encoded_row.append(val)                         # 原文

        molecule_data_index.append(encoded_row)

# ====================== 保存 CSV ======================
csv_out = os.path.join(path_out, "molecule_data_index.csv")
with open(csv_out, 'w', encoding='utf-8') as fw:
    for row in molecule_data_index:
        fw.write(",".join(row) + "\n")

# ====================== 保存 TXT ======================
txt_out = os.path.join(path_out, "molecule_data_index.txt")
with open(txt_out, 'w', encoding='utf-8') as fw:
    for row in molecule_data_index:
        fw.write(" ".join(row) + "\n")

# ====================== 保存总文件 TXT ======================
txt_out = os.path.join(path_out_in, "total_data.txt")
with open(txt_out, 'w', encoding='utf-8') as fw:
    for row in molecule_data_index:
        fw.write(" ".join(row) + "\n")

print("test3 finished")
