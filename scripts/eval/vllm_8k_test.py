import sys
import json
import argparse
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from deepscaler.rewards.math_reward import deepscaler_reward_fn
import traceback

suffix_1 = (
    " Let's think step by step and output the final answer within \\boxed{}."  # original suffix
)
suffix_2 = (
    " When answering the above question, if you're highly confident, provide concise reasoning;"
    " if uncertain, reason step-by-step clearly, avoid repetition, and always present your final answer"
    " within \\boxed{final answer}."
)
suffix_3 = ""
suffix_4 = (
    "\n\nPlease reason step by step, and put your final answer letter in \\boxed{}, like \\boxed{A}."
)

def request_model(client, prompt, seed, model):
    def single_request():
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "add_generation_prompt": True,
                "seed": seed,
            },
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            timeout=100000000,
        )
        return response
    response = single_request()
    model_response = response.choices[0].message.content.strip()

    return model_response

def main(seed):
    parser = argparse.ArgumentParser(description="vLLM performs inference and calculates pass@1 and pass@16.")
    parser.add_argument('--model', type=str, required=True, help="Path to the model ")
    parser.add_argument('--file', type=str, required=True, help="Path to the test set JSON file")
    parser.add_argument('--ports', type=str, required=True, help="List of vLLM server port numbers, separated by commas.")
    parser.add_argument('--repeat', type=int, required=True, help="The number of repetitions for each question (e.g., 64)")
    parser.add_argument('--concurrency', type=int, required=True, help="The maximum number of concurrent requests per port.")
    parser.add_argument('--output_dir', type=str, required=True, help="Output folder.")
    args = parser.parse_args()
    suffix = suffix_1

    # Construct multiple vLLM clients, each corresponding to a different port, and use a semaphore to limit the number of concurrent executions.
    ports = args.ports.split(',')
    client_list = []
    for port in ports:
        openai_api_key = "sk-xxx"
        openai_api_base = f"http://localhost:{port}/v1"
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        sem = threading.Semaphore(args.concurrency)
        client_list.append((client, sem))
    total_clients = len(client_list)
    total_concurrency = total_clients * args.concurrency
    print(f"Constructed a total of {total_clients} clients, with a total concurrency of {total_concurrency}")

    # Read the raw test data (in list format), where each element contains a "problem" and an "answer".
    with open(args.file, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # Construct a task list, with each task corresponding to a single request (repeat the same question args.repeat times).
    tasks = []   # (question_index, prompt, ground_truth, repetition_index)
    for q_idx, item in enumerate(original_data):
        prompt = item["problem"] + suffix
        prompt = prompt.strip()
        ground_truth = item["answer"]
        if q_idx == 0:
            print(f"Example question: {prompt}, example answer: {ground_truth}")
        for rep in range(args.repeat):
            tasks.append((q_idx, prompt, ground_truth, rep))

    detailed_results = {}
    for q_idx, item in enumerate(original_data):
        prompt = item["problem"] + suffix
        prompt = prompt.strip()
        ground_truth = item["answer"]
        detailed_results[q_idx] = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "responses": [None] * args.repeat,
            "correct_flags": [False] * args.repeat
        }

    total_responses = len(tasks)
    correct_count = 0

    executor = ThreadPoolExecutor(max_workers=total_concurrency)
    future_list = []
    global_index = 0

    def request_task(client, semaphore, prompt, seed, model):
        with semaphore:
            return request_model(client, prompt, seed, model)

    for task in tasks:
        q_idx, prompt, ground_truth, rep = task
        client, sem = client_list[global_index % total_clients]
        global_index += 1
        fut = executor.submit(request_task, client, sem, prompt, seed, args.model)
        future_list.append((q_idx, rep, prompt, ground_truth, fut))

    for q_idx, rep, prompt, ground_truth, fut in future_list:
        try_count = 0
        while try_count <= 10:
            try:
                model_response = fut.result()
                break
            except Exception as e:
                try_count += 1
                error_message = traceback.format_exc()
                print("Error, retrying", error_message, flush=True)
                model_response = f"Error, {error_message}"
        processed_response = model_response.replace("\n", "")
        try:
            is_correct = deepscaler_reward_fn(model_response, ground_truth)
        except Exception as e:
            is_correct = False

        detailed_results[q_idx]["responses"][rep] = processed_response
        detailed_results[q_idx]["correct_flags"][rep] = is_correct

        if is_correct:
            correct_count += 1

    pass_at_1 = correct_count / total_responses
    pass16_correct = 0
    for q_idx, result in detailed_results.items():
        question_correct = any(result["correct_flags"])
        result["is_correct"] = question_correct
        result["question_accuracy"] = sum(result["correct_flags"]) / args.repeat
        if question_correct:
            pass16_correct += 1
    total_questions = len(original_data)
    pass_at_16 = pass16_correct / total_questions

    summary_results = {
        "file": args.file,
        "model": args.model,
        "pass@1": pass_at_1,
        f"pass@{args.repeat}": pass_at_16,
        "total_responses": total_responses,
        "total_questions": total_questions
    }
    with open(args.output_dir + "/results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    detailed_results_list = []
    for q_idx in sorted(detailed_results.keys()):
        detailed_results_list.append(detailed_results[q_idx])
    with open(args.output_dir + "/results_details.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results_list, f, ensure_ascii=False, indent=2)

    print("######### Deepscaler Eval Results:")
    print(f"pass@1 = {pass_at_1:.4f}, pass@{args.repeat} = {pass_at_16:.4f}")
    print(f"Saving results to {args.output_dir}/results_summary.json and {args.output_dir}/results_details.json")

if __name__ == "__main__":
    main(None)
