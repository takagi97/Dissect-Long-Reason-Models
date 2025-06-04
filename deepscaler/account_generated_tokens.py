import json
from transformers import AutoTokenizer
import sys

tokenizer = AutoTokenizer.from_pretrained("/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/original_weights/DeepScaleR-1.5B-Preview")
file_path = "/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/original_weights/DeepScaleR-1.5B-Preview/aime/results_details.json"
with open(file_path, "r") as f:
    data = json.load(f)

total_tokens = 0
total_responses = 0

for item in data:
    responses = item["responses"]
    token_counts = [len(tokenizer.encode(resp)) for resp in responses]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    # print(f"Prompt: {item['prompt'][:30]}... -> Avg tokens per response: {avg_tokens:.2f}")
    
    total_tokens += sum(token_counts)
    total_responses += len(responses)

print("\n====== Summary ======")
print(f"Total responses: {total_responses}")
print(f"Total tokens across all responses: {total_tokens}")
print(f"Average tokens per response: {total_tokens / total_responses:.2f}")
