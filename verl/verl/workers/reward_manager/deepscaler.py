from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from deepscaler.rewards.math_reward import deepscaler_reward_fn
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        return deepscaler_reward_fn

def process_item(args):
    i, data_item, already_print_data_sources, tokenizer = args
    prompt_ids = data_item.batch['prompts']
    prompt_length = prompt_ids.shape[-1]

    valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = data_item.batch['responses'] 
    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]

    # decode
    sequences = torch.cat((valid_prompt_ids, valid_response_ids))
    sequences_str = tokenizer.decode(sequences)

    ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

    # select rm_score
    data_source = data_item.non_tensor_batch['data_source']
    compute_score_fn = _select_rm_score_fn(data_source)
    score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
    valid_response_length = int(valid_response_length.item())
     
    return i, score, valid_response_length, sequences_str


class DeepScalerRewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto, return_corr_list=False, enble_relative_length_reward=False, rlr_args={"corr_ratio": 0.5, "upper_bound": 0.5, "lower_bound": -0.5, "length_base": 1000}, enble_repetition_reward=False, rr_args={"ngram_size": 40, "max_penalty": -0.5}):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
    
        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i], already_print_data_sources, self.tokenizer) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        corr_list = []
        response_length_list = []
        response_list = []
        for i, score, valid_response_length, response in results:
            reward_tensor[i, valid_response_length - 1] = score
            corr_list.append(score)
            response_length_list.append(valid_response_length)
            response_list.append(response)

        def relative_length_reward(batch, reward_tensor, corr_list, response_length_list):
            bsz, _ = batch.batch["prompts"].shape
            uid_list = batch.non_tensor_batch['uid']
            uid2score = defaultdict(list)
            uid2all_idx = defaultdict(list)
            uid2all_len = defaultdict(list)
            for i in range(bsz):
                uid2score[uid_list[i]].append(corr_list[i])

            valid_ids = []
            for i in range(bsz):
                uid = uid_list[i]
                corr_ratio = sum(uid2score[uid]) / len(uid2score[uid])
                if corr_list[i] and corr_ratio >= rlr_args["corr_ratio"]: # 只统计对的采样结果 且 正确率大于阈值
                    uid2all_len[uid].append(response_length_list[i])
                    uid2all_idx[uid].append(i)
                    valid_ids.append(i)
            
            for i in valid_ids:
                mean_length = sum(uid2all_len[uid_list[i]]) / len(uid2all_len[uid_list[i]])
                curr_length = response_length_list[i]
                length_reward = (mean_length - curr_length) / rlr_args["length_base"]
                if length_reward > rlr_args["upper_bound"]:
                    length_reward = rlr_args["upper_bound"]
                if length_reward < rlr_args["lower_bound"]:
                    length_reward = rlr_args["lower_bound"]
                reward_tensor[i, curr_length - 1] += length_reward

        if enble_relative_length_reward:
            relative_length_reward(data, reward_tensor, corr_list, response_length_list)


        def repetition_reward(reward_tensor, corr_list, response_length_list, response_list, repetition_penalty_list):
            # copied from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
            all_idx = []
            
            for i, if_corr in enumerate(corr_list):
                if not if_corr: # 只统计错的采样结果
                    all_idx.append(i)
            
            # Source: https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
            def zipngram(text: str, ngram_size: int):
                words = text.lower().split()
                return zip(*[words[i:] for i in range(ngram_size)])

            for i in all_idx:
                response = response_list[i]
                ngrams = set()
                total = 0
                for ng in zipngram(response, rr_args["ngram_size"]):
                    ngrams.add(ng)
                    total += 1
                if total == 0:
                    repetition_penalty_reward = 0
                else:
                    scaling = 1 - len(ngrams) / total
                    repetition_penalty_reward = scaling * rr_args["max_penalty"]
                curr_length = response_length_list[i]
                reward_tensor[i, curr_length - 1] += repetition_penalty_reward
                if repetition_penalty_reward != 0:
                    repetition_penalty_list[i] = True

        repetition_penalty_list = [False for _ in range(len(corr_list))]
        if enble_repetition_reward:
            repetition_reward(reward_tensor, corr_list, response_length_list, response_list, repetition_penalty_list)


        if return_corr_list:
            return reward_tensor, corr_list, repetition_penalty_list
        else:
            return reward_tensor

