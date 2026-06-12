import os

# ================================
# 1. 读取 result.txt 中的 embedding
# ================================
def load_embeddings(result_path):
    embeddings = {}
    with open(result_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 第一行：节点数 和 维度
    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"result.txt 第一行格式错误: {lines[0]}")
    try:
        num_nodes = int(header[0])
        emb_dim = int(header[1])
    except Exception as e:
        raise ValueError(f"无法解析第一行的节点数和维度: {lines[0]}") from e

    # 后面的每一行：变量名 + embedding
    for idx, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) != emb_dim + 1:
            raise ValueError(
                f"result.txt 第 {idx} 行维度不匹配：期望 {emb_dim}，实际 {len(parts)-1}\n"
                f"行内容: {line}"
            )
        key = parts[0]  # 变量名
        vec = [float(x) for x in parts[1:]]
        embeddings[key] = vec

    print(f"[load_embeddings] 从 {result_path} 读取到 {len(embeddings)} 个节点，维度 = {emb_dim}")
    if len(embeddings) != num_nodes:
        print(f"⚠ 警告：header 中的节点数 = {num_nodes}，但实际读取 = {len(embeddings)}")

    return embeddings, emb_dim


# ================================
# 2. 将 training/testing_total 转为 embedding
# ================================
def convert_data(data_path, output_path, embeddings, emb_dim, num_vars=6):
    """
    data_path: training_data_total.txt / testing_data_total.txt
    每行：ID v1 v2 v3 v4 v5 v6
    输出：v1..v6 的 embedding 拼接（不含 ID）
    """
    with open(data_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        line_count = 0
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            parts = line.split()

            if len(parts) < 1 + num_vars:
                raise ValueError(
                    f"[convert_data] 文件 {data_path} 第 {line_count} 行列数不足："
                    f"期望 ≥ {1+num_vars} 列，实际 {len(parts)}。\n行内容: {line}"
                )

            sample_id = parts[0]
            var_tokens = parts[1:1+num_vars]

            emb_concat = []
            for j, token in enumerate(var_tokens):
                key = token
                if key not in embeddings:
                    raise KeyError(
                        f"[convert_data] 在 {data_path} 第 {line_count} 行，第 {j+2} 列：\n"
                        f"变量值 '{key}' 在 result embedding 中找不到！\n"
                        f"请检查 result.txt 是否包含这一项。"
                    )
                emb_vec = embeddings[key]
                emb_concat.extend(emb_vec)

            if len(emb_concat) != num_vars * emb_dim:
                raise ValueError(
                    f"[convert_data] 第 {line_count} 行 embedding 维度错误："
                    f"期望 {num_vars * emb_dim}，实际 {len(emb_concat)}。"
                )

            f_out.write(" ".join(str(x) for x in emb_concat) + "\n")

    print(f"[convert_data] 已从 {data_path} 生成 {line_count} 行 embedding → {output_path}")
    return line_count  # ⬅ 返回生成的行数


# ================================
# 主程序
# ================================
result_path = "./data/test6/result.txt"
train_path  = "./data/test1/training_data_total.txt"
test_path   = "./data/test1/testing_data_total.txt"

train_out = "./data/test7/training_data_embedding.txt"
test_out  = "./data/test7/testing_data_embedding.txt"

# 1) 读取 result embedding
embeddings, emb_dim = load_embeddings(result_path)

# 2) 生成 training / testing embedding 文件
train_rows = convert_data(train_path, train_out, embeddings, emb_dim, num_vars=6)
test_rows  = convert_data(test_path, test_out, embeddings, emb_dim, num_vars=6)

print("============================================")
print("Embedding 文件维度检查：")
print(f"训练集：样本数 = {train_rows}，每行维度 = {6 * emb_dim}")
print(f"测试集：样本数 = {test_rows}，每行维度 = {6 * emb_dim}")
print("============================================")

print("✅ 全部处理完成！")
