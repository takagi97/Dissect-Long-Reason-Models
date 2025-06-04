import glob
import json
import re

# 定义 JSON 文件匹配的路径模式 arithmetic
pattern = "/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/r1/train/deepscaler/checkpoints/deepscaler/4.20.8k-A100-bsz64-reserve0_512_after_positive/global_step_*/aime-8k/results_summary.json"
print(pattern)
# 查找所有符合模式的 JSON 文件
json_files = glob.glob(pattern)

# 定义一个函数，从文件路径中提取 global_step 后面的数字
def extract_step(file_path):
    match = re.search(r'global_step_(\d+)', file_path)
    if match:
        return int(match.group(1))
    else:
        return float('inf')  # 如果未找到数字，则排到最后

# 按照 global_step 的数字升序排序 JSON 文件路径
json_files = sorted(json_files, key=extract_step)

# 遍历每个 JSON 文件，读取 pass@1 分数并打印
for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 从路径中提取 step 数字
        step = extract_step(json_file)
        # 获取 JSON 中的 pass@1 分数
        score = data.get("pass@1")
        # 按格式打印：step 数字 和分数
        print(f"step {step} {score}")
