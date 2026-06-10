import json
import argparse
from pathlib import Path
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn



def parse_gsm8k(data_path):
    # GSM8K is stored as JSONL, so each line is one independent sample.
    items = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            items.append(item)
    return items

def form_prompt(prompt_path, items):
    # Prompt templates are used as one whole string, not line by line.
    with open(prompt_path) as f:
        prompt = f.read()
    prompts = []
    answers = []
    for item in items:
        question = item["question"]
        answer = item["answer"]
        result = prompt.replace("{question}", question)
        prompts.append(result)
        answers.append(answer)
    return prompts, answers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_path", type=str)
    parser.add_argument("data_path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("output_path", type=str, default="outputs/section3")
    parser.add_argument("model_name", type=str, default="allenai/OLMo-2-0425-1B")
    args = parser.parse_args()

    # Build paths from the repo root so this script still works
    # even if it is launched from a different working directory.
    root = Path(__file__).resolve().parents[2]

    prompt_path = root / args.prompt_path
    data_path = root / args.data_path
    output_path = root / args.output_path
    model_name = args.model_name

    # output_path is treated as a directory that holds one result file per prompt.
    output_path.mkdir(parents=True, exist_ok=True)

    items = parse_gsm8k(data_path)
    prompts, answers = form_prompt(prompt_path, items)

    # Start a local vLLM server instead of loading the model directly in this script.
    print(f"Running prompt: {prompt_path.name}")
    server = VLLMServer(model_id=model_name)
    server.start()

    #r1_zero needs to stop generation after </answer> token
    sampling_params = {
        "temperature": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": 42,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True
    }
    if prompt_path.stem == "question_only":
        del sampling_params["stop"]
        del sampling_params["include_stop_str_in_output"]

    completions = server.generate_completions(
        prompts,
        sampling_params,
        batch_size=16
    )

    # The reward function expects plain response strings, not VLLM objects.
    responses = []
    for completion in completions:
        response = completion.text
        responses.append(response)

    rewards = []
    summary = {}
    for i in range(len(responses)):
        # Score each sample independently, then keep only the scalar total reward.
        if prompt_path.stem == "question_only":
            reward = question_only_reward_fn(responses[i], answers[i])
        else:
            reward = r1_zero_reward_fn(responses[i], answers[i])
        rewards.append(reward["reward"])

        #According to assignments, three forms of rewards needs to be reported
        if reward["format_reward"] == 1 and reward["answer_reward"] == 1:
            summary["correct"] = summary.get("correct", 0) + 1
        elif reward["format_reward"] == 1 and reward["answer_reward"] == 0:
            summary["format_only"] = summary.get("format_only", 0) + 1
        else:
            summary["unformatted"] = summary.get("unformatted", 0) + 1
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average rewards: {avg_reward:.4f}")
    print("Summary:")
    print(summary)
    server.stop()

    prompt_name = prompt_path.stem
    # Use the prompt filename in the output name so different prompt variants do not overwrite each other.
    result_path = output_path / f"{prompt_name}.jsonl"
    
    print(f"File saving to {result_path}")
    with open(result_path, "w") as f:
        # Save one sample per line so the file can be read back as JSONL.
        for i in range(len(prompts)):
            result = {
                "prompt": prompts[i],
                "answer": answers[i],
                "response": responses[i],
                "reward": rewards[i]
            }
            json.dump(result, f)
            f.write("\n")
    print("File saved")
    
if __name__ == "__main__":
    main()
