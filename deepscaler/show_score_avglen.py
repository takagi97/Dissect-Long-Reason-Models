import glob
import json
import os
import re
from transformers import AutoTokenizer

# === 1. 配置 ===
# ① 结果文件路径模式（只写到 results_summary.json）
# PATTERN = (
#     "/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/r1/train/deepscaler/checkpoints/deepscaler/5.3.DeepScaleR-1.5B-Preview-continue_24k-SLR_all_corr_ratio0.75-bound0.5_1-length_base500-max_response8k-noKL_entropy/global_step_*/aime/results_summary.json"
# )
PATTERN = (
    "/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/r1/train/deepscaler/checkpoints/deepscaler/5.7.DeepScaleR-1.5B-Preview-continue_24k-SLR_all_corr_ratio0.75-bound0.5_1-length_base500-max_response4k-noKL_entropy/global_step_320/*/results_summary.json"
)
print(PATTERN)

# ② tokenizer 权重目录（一次加载即可）
TOKENIZER_DIR = (
    "/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/"
    "original_weights/DeepScaleR-1.5B-Preview"
)

# === 2. 函数 ===
def extract_step(path: str) -> int:
    """从路径中提取 global_step 的数字⽤于排序/打印。"""
    m = re.search(r"global_step_(\d+)", path)
    return int(m.group(1)) if m else -1


def average_length(details_path: str, tokenizer) -> float:
    """
    读取 results_details.json，统计所有 responses 的平均 token 数。
    若⽂件不存在或数据为空则返回 float('nan')。
    """
    if not os.path.exists(details_path):
        return float("nan")

    with open(details_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_toks = 0
    total_resp = 0
    for item in data:
        resp_list = item.get("responses", [])
        # add_special_tokens=False 可稍微快⼀点，也避免计算 BOS/EOS
        tok_counts = [len(tokenizer.encode(r, add_special_tokens=False)) for r in resp_list]
        total_toks += sum(tok_counts)
        total_resp += len(tok_counts)

    return (total_toks / total_resp) if total_resp else float("nan")


# === 3. 主流程 ===
def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    json_files = sorted(glob.glob(PATTERN), key=extract_step)

    print(f"{'step':>8}  {'pass@1':>8}  {'avg_len':>10}")
    print("-" * 30)

    for summary_path in json_files:
        step = extract_step(summary_path)

        # 读取 pass@1
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        pass1 = summary.get("pass@1", float("nan"))

        # 计算平均输出⻓度
        details_path = summary_path.replace("results_summary.json", "results_details.json")
        avg_len = average_length(details_path, tokenizer)

        print(f"{step:8d}  {pass1:8.4f}  {avg_len:10.2f}")


if __name__ == "__main__":
    main()
