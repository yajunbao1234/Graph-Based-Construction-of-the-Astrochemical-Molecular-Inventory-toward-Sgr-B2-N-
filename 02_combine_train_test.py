#####把训练数据和测试数据放在一起
import os

path_in = './data/test1'
path_out = './data/test2'

# 定义源文件和目标文件的路径
source_file_1 = os.path.join(path_in, "training_data.txt")
source_file_2 = os.path.join(path_in, "testing_data.txt")

destination_file = os.path.join(path_out, "training_testing_combined.txt")

# 打开源文件和目标文件
with open(source_file_1, 'r', encoding='utf-8') as file1, \
     open(source_file_2, 'r', encoding='utf-8') as file2, \
     open(destination_file, 'w', encoding='utf-8') as combined_file:

    # 读取第一个文件的内容并写入目标文件
    content1 = file1.read()
    combined_file.write(content1)

    # 读取第二个文件的内容并写入目标文件
    content2 = file2.read()
    combined_file.write(content2)

print("test2 finished")