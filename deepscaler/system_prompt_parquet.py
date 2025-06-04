import pandas as pd
import argparse


def process_prompt_field(prompt_list: list) -> list:
    if not prompt_list or 'content' not in prompt_list[0]:
        return prompt_list

    content = prompt_list[0]['content']
    old_suffix = (
        " Let's think step by step and output the final answer within \\boxed{}."  # original suffix
    )
    new_suffix = (
        " When answering the above question, if you're highly confident, provide concise reasoning;"
        " if uncertain, reason step-by-step clearly, avoid repetition, and always present your final answer"
        " within \\boxed{final answer}."
    )
    
    if content.endswith(old_suffix):
        # Remove the old suffix and append the new one
        content = content[: -len(old_suffix)] + new_suffix
    else:
        # Fallback: replace all occurrences
        content = content.replace(old_suffix, new_suffix)

    prompt_list[0]['content'] = content
    return prompt_list


def main(input_path: str, output_path: str):
    # Read the parquet file
    df = pd.read_parquet(input_path)

    # Ensure 'prompt' column exists
    if 'prompt' not in df.columns:
        raise KeyError("The input parquet file does not contain a 'prompt' column.")

    # Apply transformation
    df['prompt'] = df['prompt'].apply(process_prompt_field)

    # Write out the modified parquet
    df.to_parquet(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == '__main__':
    input_path = "/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/r1/train/deepscaler/deepscaler/data/myy-data/train.dedup_long.parquet"
    output_path = "/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/jerrymu/r1/train/deepscaler/deepscaler/data/myy-data/train.dedup_long.sys_v2.parquet"
    main(input_path, output_path)
