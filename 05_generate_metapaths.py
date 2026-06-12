######## 得到随机游走路径

import numpy as np
import random
import time
import tqdm
import dgl
import sys
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt

start = time.time()
print('Running time: %s Seconds' % (start))

num_walks_per_node = 5  # 20
walk_length = 10         # 8

####此处需要修改，注意后续embedding results存储位置
path_in = './data/test4'
path_out = './data/test5'
####此处需要修改，注意后续embedding results存储位置

# 8 个变量名称（当前只用 v1-v6）
nodes = ["v1", "v2", "v3", "v4", "v5", "v6"]

# 每个变量的 ids, names, dict
v_ids = {n: [] for n in nodes}
v_names = {n: [] for n in nodes}
v_ids_names = {n: {} for n in nodes}

# 打开文件
files = {n: open(os.path.join(path_in, f"{n}.txt"), encoding="ISO-8859-1") for n in nodes}

for n in nodes:
    f = files[n]
    while True:
        line = f.readline()
        if not line:
            break

        parts = line.strip().split()
        identity = int(parts[0])

        # v8 特殊：value 可能有空格 → 保留所有剩余内容（当前没有 v8，这里保持原写法不动）
        if n == "v8":
            value = "".join(parts[1:])
        else:
            value = parts[1]

        v_ids[n].append(identity)
        v_names[n].append(value)
        v_ids_names[n][identity] = value

# 关闭文件
for f in files.values():
    f.close()

v1_ids, v1_names, v1_ids_names = v_ids["v1"], v_names["v1"], v_ids_names["v1"]
v2_ids, v2_names, v2_ids_names = v_ids["v2"], v_names["v2"], v_ids_names["v2"]
v3_ids, v3_names, v3_ids_names = v_ids["v3"], v_names["v3"], v_ids_names["v3"]
v4_ids, v4_names, v4_ids_names = v_ids["v4"], v_names["v4"], v_ids_names["v4"]
v5_ids, v5_names, v5_ids_names = v_ids["v5"], v_names["v5"], v_ids_names["v5"]
v6_ids, v6_names, v6_ids_names = v_ids["v6"], v_names["v6"], v_ids_names["v6"]



def extract_selfies_tokens(s):
    """
    从 SELFIES 字符串中提取 token，例如 [C][C][=O][Branch1]...
    不依赖外部库，只根据方括号分割
    """
    tokens = []
    cur = ""
    inside = False
    for ch in s:
        if ch == '[':
            inside = True
            cur = "["
        elif ch == ']':
            cur += "]"
            inside = False
            tokens.append(cur)
            cur = ""
        else:
            if inside:
                cur += ch
    return set(tokens)

# 每个 v6 节点（identity）对应一个 SELFIES token 集合
v6_token_sets = {
    idx: extract_selfies_tokens(v6_ids_names[idx])
    for idx in v6_ids
}

# 构建 v6-v6 功能团重叠边：只要有至少一个 token 相同就连边（双向）
v6_v6_src = []
v6_v6_dst = []

v6_list = v6_ids
N6 = len(v6_list)

for i in range(N6):
    for j in range(i + 1, N6):
        a = v6_list[i]
        b = v6_list[j]
        if len(v6_token_sets[a] & v6_token_sets[b]) > 0:
            # a -> b
            v6_v6_src.append(a)
            v6_v6_dst.append(b)
            # b -> a
            v6_v6_src.append(b)
            v6_v6_dst.append(a)

edge_src = {}
edge_dst = {}

# 生成所有 1 ≤ i < j ≤ 8 的组合（当前只用到 v1-v6 之间的若干对）
pairs = [(i, j) for i in range(1, 8) for j in range(i + 1, 8)]

for i, j in pairs:
    key = f"v{i}_v{j}"
    edge_src[key] = []
    edge_dst[key] = []

for i, j in pairs:
    key = f"v{i}_v{j}"
    filename = f"v{i}_v{j}.txt"
    path = os.path.join(path_in, filename)

    # 某些 v*_v* 文件在当前 6 变量设置下可能不存在，如果不存在就跳过
    if not os.path.exists(path):
        continue

    with open(path, "r") as f:
        for line in f:
            a, b = line.strip().split('\t')
            edge_src[key].append(int(a))
            edge_dst[key].append(int(b))

# 这里只取你原来实际用到的那几组边
v1_v2_src = edge_src.get("v1_v2", []); v1_v2_dst = edge_dst.get("v1_v2", [])
v1_v3_src = edge_src.get("v1_v3", []); v1_v3_dst = edge_dst.get("v1_v3", [])
v1_v4_src = edge_src.get("v1_v4", []); v1_v4_dst = edge_dst.get("v1_v4", [])
v1_v5_src = edge_src.get("v1_v5", []); v1_v5_dst = edge_dst.get("v1_v5", [])
v1_v6_src = edge_src.get("v1_v6", []); v1_v6_dst = edge_dst.get("v1_v6", [])

v2_v3_src = edge_src.get("v2_v3", []); v2_v3_dst = edge_dst.get("v2_v3", [])
v2_v4_src = edge_src.get("v2_v4", []); v2_v4_dst = edge_dst.get("v2_v4", [])
v2_v5_src = edge_src.get("v2_v5", []); v2_v5_dst = edge_dst.get("v2_v5", [])
v2_v6_src = edge_src.get("v2_v6", []); v2_v6_dst = edge_dst.get("v2_v6", [])

v3_v4_src = edge_src.get("v3_v4", []); v3_v4_dst = edge_dst.get("v3_v4", [])
v3_v5_src = edge_src.get("v3_v5", []); v3_v5_dst = edge_dst.get("v3_v5", [])
v3_v6_src = edge_src.get("v3_v6", []); v3_v6_dst = edge_dst.get("v3_v6", [])

v4_v5_src = edge_src.get("v4_v5", []); v4_v5_dst = edge_dst.get("v4_v5", [])
v4_v6_src = edge_src.get("v4_v6", []); v4_v6_dst = edge_dst.get("v4_v6", [])

v5_v6_src = edge_src.get("v5_v6", []); v5_v6_dst = edge_dst.get("v5_v6", [])

# 构建异构图
hg = dgl.heterograph({
    ('v1', 'v1v2', 'v2'): (v1_v2_src, v1_v2_dst),
    ('v2', 'v2v1', 'v1'): (v1_v2_dst, v1_v2_src),

    ('v1', 'v1v3', 'v3'): (v1_v3_src, v1_v3_dst),
    ('v3', 'v3v1', 'v1'): (v1_v3_dst, v1_v3_src),

    ('v1', 'v1v4', 'v4'): (v1_v4_src, v1_v4_dst),
    ('v4', 'v4v1', 'v1'): (v1_v4_dst, v1_v4_src),

    ('v1', 'v1v5', 'v5'): (v1_v5_src, v1_v5_dst),
    ('v5', 'v5v1', 'v1'): (v1_v5_dst, v1_v5_src),

    ('v1', 'v1v6', 'v6'): (v1_v6_src, v1_v6_dst),
    ('v6', 'v6v1', 'v1'): (v1_v6_dst, v1_v6_src),

    ('v2', 'v2v3', 'v3'): (v2_v3_src, v2_v3_dst),
    ('v3', 'v3v2', 'v2'): (v2_v3_dst, v2_v3_src),

    ('v2', 'v2v4', 'v4'): (v2_v4_src, v2_v4_dst),
    ('v4', 'v4v2', 'v2'): (v2_v4_dst, v2_v4_src),

    ('v2', 'v2v5', 'v5'): (v2_v5_src, v2_v5_dst),
    ('v5', 'v5v2', 'v2'): (v2_v5_dst, v2_v5_src),

    ('v2', 'v2v6', 'v6'): (v2_v6_src, v2_v6_dst),
    ('v6', 'v6v2', 'v2'): (v2_v6_dst, v2_v6_src),

    ('v3', 'v3v4', 'v4'): (v3_v4_src, v3_v4_dst),
    ('v4', 'v4v3', 'v3'): (v3_v4_dst, v3_v4_src),

    ('v3', 'v3v5', 'v5'): (v3_v5_src, v3_v5_dst),
    ('v5', 'v5v3', 'v3'): (v3_v5_dst, v3_v5_src),

    ('v3', 'v3v6', 'v6'): (v3_v6_src, v3_v6_dst),
    ('v6', 'v6v3', 'v3'): (v3_v6_dst, v3_v6_src),

    ('v4', 'v4v5', 'v5'): (v4_v5_src, v4_v5_dst),
    ('v5', 'v5v4', 'v4'): (v4_v5_dst, v4_v5_src),

    ('v4', 'v4v6', 'v6'): (v4_v6_src, v4_v6_dst),
    ('v6', 'v6v4', 'v4'): (v4_v6_dst, v4_v6_src),

    ('v5', 'v5v6', 'v6'): (v5_v6_src, v5_v6_dst),
    ('v6', 'v6v5', 'v5'): (v5_v6_dst, v5_v6_src),

    # ⭐ 新增：v6-v6 功能团相同即可游走的边（双向）
    ('v6', 'v6v6', 'v6'): (v6_v6_src, v6_v6_dst),
})


def generate_metapath():
    output_path = open(os.path.join(path_out, 'output_path.txt'), 'w', encoding="ISO-8859-1")

    # 名字映射表（只写一次）
    name_dict = {
        'v1': v1_ids_names, 'v2': v2_ids_names, 'v3': v3_ids_names, 'v4': v4_ids_names,
        'v5': v5_ids_names, 'v6': v6_ids_names
    }

    # =============================================================
    # 1) 第一条随机规则 —— 可随便改 mp1，下面不用动
    # =============================================================
    mp1 = ['v1v2', 'v2v3', 'v3v4', 'v4v3', 'v3v2', 'v2v1']
    L1 = len(mp1)
    dst1 = [e[-2:] for e in mp1]

    for v_idx in v1_ids:
        traces, _ = dgl.sampling.random_walk(
            hg, [v_idx], metapath=mp1 * walk_length
        )
        for tr in traces:
            tr = tr.tolist()
            outline = v1_ids_names[tr[0]]
            for i in range(1, len(tr)):
                outline += " " + name_dict[dst1[(i - 1) % L1]][tr[i]]
            output_path.write(outline + "\n")
    print("第一条随机游走完成")

    # =============================================================
    # 2) 第二条
    # =============================================================
    mp2 = ['v2v3', 'v3v4', 'v4v5', 'v5v6', 'v6v6', 'v6v5', 'v5v4', 'v4v3', 'v3v2']
    L2 = len(mp2)
    dst2 = [e[-2:] for e in mp2]

    for v_idx in v2_ids:
        traces, _ = dgl.sampling.random_walk(
            hg, [v_idx], metapath=mp2 * walk_length
        )
        for tr in traces:
            tr = tr.tolist()
            outline = v2_ids_names[tr[0]]
            for i in range(1, len(tr)):
                outline += " " + name_dict[dst2[(i - 1) % L2]][tr[i]]
            output_path.write(outline + "\n")
    print("第二条随机游走完成")

    # =============================================================
    # 3) 第三条
    # =============================================================
    mp3 = ['v3v2', 'v2v3']
    L3 = len(mp3)
    dst3 = [e[-2:] for e in mp3]

    for v_idx in v3_ids:
        traces, _ = dgl.sampling.random_walk(
            hg, [v_idx], metapath=mp3 * walk_length
        )
        for tr in traces:
            tr = tr.tolist()
            outline = v3_ids_names[tr[0]]
            for i in range(1, len(tr)):
                outline += " " + name_dict[dst3[(i - 1) % L3]][tr[i]]
            output_path.write(outline + "\n")
    print("第三条随机游走完成")

    # =============================================================
    # 4) 第四条
    # =============================================================
    mp4 = ['v4v3', 'v3v2', 'v2v3', 'v3v4']
    L4 = len(mp4)
    dst4 = [e[-2:] for e in mp4]

    for v_idx in v4_ids:
        traces, _ = dgl.sampling.random_walk(
            hg, [v_idx], metapath=mp4 * walk_length
        )
        for tr in traces:
            tr = tr.tolist()
            outline = v4_ids_names[tr[0]]
            for i in range(1, len(tr)):
                outline += " " + name_dict[dst4[(i - 1) % L4]][tr[i]]
            output_path.write(outline + "\n")
    print("第四条随机游走完成")

    # =============================================================
    # 5) 第五条
    # =============================================================
    mp5 = ['v5v2', 'v2v6', 'v6v6', 'v6v2', 'v2v5']
    L5 = len(mp5)
    dst5 = [e[-2:] for e in mp5]

    for v_idx in v5_ids:
        traces, _ = dgl.sampling.random_walk(
            hg, [v_idx], metapath=mp5 * walk_length
        )
        for tr in traces:
            tr = tr.tolist()
            outline = v5_ids_names[tr[0]]
            for i in range(1, len(tr)):
                outline += " " + name_dict[dst5[(i - 1) % L5]][tr[i]]
            output_path.write(outline + "\n")
    print("第五条随机游走完成")

    # =============================================================
    # 6) 第六条
    # =============================================================
    mp6 = ['v6v5', 'v5v2', 'v2v5', 'v5v6']
    L6 = len(mp6)
    dst6 = [e[-2:] for e in mp6]

    for v_idx in v6_ids:
        traces, _ = dgl.sampling.random_walk(
            hg, [v_idx], metapath=mp6 * walk_length
        )
        for tr in traces:
            tr = tr.tolist()
            outline = v6_ids_names[tr[0]]
            for i in range(1, len(tr)):
                outline += " " + name_dict[dst6[(i - 1) % L6]][tr[i]]
            output_path.write(outline + "\n")
    print("第六条随机游走完成")

    output_path.close()


generate_metapath()

end = time.time()
print('Running time: %s Seconds' % (end))
print('Running time: %s Seconds' % (end - start))

print('test5 finished')
