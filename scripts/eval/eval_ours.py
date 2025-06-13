import json, re
from pathlib import Path
from threading import Thread
from queue import Queue
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rllm.rewards.code_reward import lcb_check_correctness_v2, extract_code_from_model
from data.scripts.prompt_utils import get_stage2_prompt

# ── CONFIG ──────────────────────────────────────────────────────────────────────
MODEL1_PATH = "/root/autodl-tmp/ckpt/ours/sft/checkpoint-396"
MODEL2_PATH = "/home/ducati/projects/finalproject/rllm/checkpoints/deepcoder/1.5b-4k-gae-code/actor/global_step_40"
JSON_DATA   = "/home/ducati/projects/finalproject/rllm/data/test_livecodebench.json"
OUT_PATH    = "/root/autodl-tmp/outputs/test_output_result.json"
BATCH_SIZE  = 8

DEV1 = torch.device("cuda:0")
DEV2 = torch.device("cuda:0")

# BF16 autocast context manager
AUTOCAST = torch.cuda.amp.autocast
# ── END CONFIG ──────────────────────────────────────────────────────────────────

def get_stage1_prompt(input_prompt):
    if not isinstance(input_prompt, str):
        input_prompt = input_prompt[0]['content'] if isinstance(input_prompt, list) else str(input_prompt)
    problem = re.search(r'passes all tests\.(.*?)### Answer', input_prompt, re.DOTALL)
    if problem:
        problem = problem.group(1).strip()
    else:
        print(f"Input prompt: {input_prompt}")
        raise ValueError("Could not find problem description in the input prompt.")
    prompt_s1 = f"""
    You are an expert Python programmer. You will be given a question (problem specification), and you have to provide an incomplete code snippet that contains the overall pipeline (in the main block) and masked sub-tasks (functions with docstrings but no implementation).
    More specifically, you have to provide a solution that contains the following parts:
    1. The main block that contains the overall pipeline of the solution. It should start with if __name__ == \"__main__\": and contain the main logic of the solution.
    2. You should define functions for each sub-task that is required to solve the problem. Each function should have a docstring that describes the task, but the implementation should be masked with \"# YOUR CODE HERE\".
    3. The code should be syntactically correct and runnable, but the implementation of the functions should be missing.
    
    An example of the code snippet is as follows:
    
    def read_test_cases():
        \"\"\"
        Read the number of test cases and for each test case, read n, k, and the string s.
        
        Returns:
            list of tuples: Each tuple contains (n, k, s) for each test case.
        \"\"\"
        # YOUR CODE HERE

    def process_test_case(n, k, s):
        \"\"\"
        Process a single test case to find the minimum operations to turn all B's to W's.
        
        Args:
            n: Length of the string.
            k: The number of consecutive cells to flip in each operation.
            s: The string representing the cells.
        
        Returns:
            int: The minimum number of operations needed.
        \"\"\"
        # YOUR CODE HERE

    def print_results(results):
        \"\"\"
        Print all results for each test case.
        
        Args:
            results: List of integers representing the results for each test case.
        \"\"\"
        # YOUR CODE HERE

    if __name__ == "__main__":
        test_cases = read_test_cases()

        results = []
        for case in test_cases:
            n, k, s = case
            res = process_test_case(n, k, s)
            results.append(res)
        
        print_results(results)
    
    Notice that the above code is an example of an incomplete code snippet that contains the overall pipeline and masked sub-tasks, but it is not the solution to the problem, and you should only refer to it as an example of the format of the code snippet you should provide.
    Your implementation should begin with ```python and end with ``` to indicate that it is a code snippet. You should not include anything else in your response.
    The problem is as follows:
    
    {problem}
    """
    return prompt_s1

def get_stage2_prompt_from_raw_output(input_prompt, stage1_output):
    problem = re.search(r'passes all tests\.(.*?)### Answer', input_prompt, re.DOTALL)
    if problem:
        problem = problem.group(1).strip()
    else:
        raise ValueError("Could not find problem description in the input prompt.")
    incomplete_code = re.search(r'```python(.*?)```', stage1_output, re.DOTALL)
    if incomplete_code:
        incomplete_code = incomplete_code.group(1).strip()
    else:
        incomplete_code = stage1_output.strip()
    
    prompt_s2 = get_stage2_prompt(problem, incomplete_code)
    return prompt_s2

class PromptDataset(Dataset):
    def __init__(self, json_path: str):
        self.items = json.load(open(json_path, "r", encoding="utf-8"))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]["prompt"]

def collate_prompts(batch: List[str]) -> List[str]:
    return batch

def main():
    # — Load tokenizers & models in bf16 —
    tok1 = AutoTokenizer.from_pretrained(MODEL1_PATH, trust_remote_code=True)
    tok1.padding_side = "left"
    tok2 = AutoTokenizer.from_pretrained(MODEL2_PATH, trust_remote_code=True)
    tok2.padding_side = "left"

    model1 = (AutoModelForCausalLM
              .from_pretrained(MODEL1_PATH, trust_remote_code=True)
              .to(DEV1, dtype=torch.bfloat16)
              .eval())
    model2 = (AutoModelForCausalLM
              .from_pretrained(MODEL2_PATH, trust_remote_code=True)
              .to(DEV2, dtype=torch.bfloat16)
              .eval())

    # — Prepare data loader —
    ds = PromptDataset(JSON_DATA)
    #for testing, only load the first 48 data samples
    ds.items = ds.items[:48]
    loader = DataLoader(ds, batch_size=BATCH_SIZE,
                        collate_fn=collate_prompts, shuffle=False)

    # — Queues for pipelining —
    stage1_q = Queue(maxsize=2)   # holds (batch_idx, orig_prompts, stage1_outs)
    results = [None] * len(ds)    # to collect final outputs

    def stage1_worker():
        for batch_idx, orig_prompts in enumerate(loader):
            # build prompts for stage 1
            s1_prompts = [get_stage1_prompt(p[0]['content']) for p in orig_prompts]
            if batch_idx < 5:
                print(f"Stage 1 prompts for item 1 in batch {batch_idx}: {s1_prompts[0]}")
            with AUTOCAST(dtype=torch.bfloat16):
                enc1 = tok1(
                    s1_prompts,
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt",
                ).to(DEV1)
                out1_ids = model1.generate(**enc1, max_new_tokens=4096)
                prompt_len = enc1["input_ids"].shape[1]
                out1_ids = out1_ids[:, prompt_len:]  # remove input part
            s1_outs = tok1.batch_decode(out1_ids, skip_special_tokens=True)
            if batch_idx < 5:
                print(f"Output of stage 1 for item 1 in batch {batch_idx}: {s1_outs[0]}")
            stage1_q.put((batch_idx, orig_prompts, s1_outs))
        # signal done
        stage1_q.put(None)

    def stage2_worker():
        while True:
            item = stage1_q.get()
            if item is None:
                break
            batch_idx, orig_prompts, s1_outs = item
            # build prompts for stage 2
            s2_prompts = [get_stage2_prompt_from_raw_output(orig_prompt[0]['content'], o) for (orig_prompt, o) in zip(orig_prompts, s1_outs)]
            if batch_idx < 5:
                print(f"Stage 2 prompts for item 1 in batch {batch_idx}: {s2_prompts[0]}")
            with AUTOCAST(dtype=torch.bfloat16):
                enc2 = tok2(
                    s2_prompts,
                    padding=True,
                    truncation=True,
                    max_length=4096,
                    return_tensors="pt",
                ).to(DEV2)
                out2_ids = model2.generate(**enc2, max_new_tokens=4096)
                # remove input part
                prompt_len = enc2["input_ids"].shape[1]
                out2_ids = out2_ids[:, prompt_len:]
            
            s2_outs = tok2.batch_decode(out2_ids, skip_special_tokens=True)

            if batch_idx < 5:
                print(f"Output of stage 2 for item 1 in batch {batch_idx}: {s2_outs[0]}")
            # store results in correct order
            for i, out in enumerate(s2_outs):
                # compute global index
                global_idx = batch_idx * BATCH_SIZE + i
                if global_idx < len(results):
                    results[global_idx] = out

    # — Launch pipeline threads —
    t1 = Thread(target=stage1_worker)
    t2 = Thread(target=stage2_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # — Write out final list —
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump([
            {"orig": ds.items[i]["prompt"], "stage2": results[i]}
            for i in range(len(results))
        ], f, ensure_ascii=False, indent=2)

    print(f"Done. Results written to {OUT_PATH}")


def compute_acc():
    START_PATTERN = "### Answer"
    BANNED_PATTERN = "```python\nclass Solution:\n"
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)

    with open(JSON_DATA, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    if len(samples) != len(preds):
        print(f"Warning: number of samples ({len(samples)}) does not match number of predictions ({len(preds)}).")
        if len(samples) < len(preds):
            preds = preds[:len(samples)]
        else:
            samples = samples[:len(preds)]
        print(f"Using {len(samples)} samples and predictions.")

    num_correct = 0
    num_total = 0
    for pred, sample in zip(preds, samples):
        if BANNED_PATTERN in pred["orig"][0]["content"]:
            continue

        response: str = pred["stage2"]
        if START_PATTERN in response:
            # delete all text before START_PATTERN
            pos = response.index(START_PATTERN)
            response = response[pos + len(START_PATTERN):].strip()
        
        num_total += 1
        try:
            code = extract_code_from_model(response)
            tests = json.loads(sample["reward_model"]["ground_truth"])
            verdict, metadata = lcb_check_correctness_v2(tests, code)
            print("=" * 50)
            print("ORIGINAL PROMPT:")
            print(sample["prompt"][0]["content"])
            print("#" * 50)
            print("STAGE 2 OUTPUT:")
            print(code)
            print("VERDICT:", verdict, metadata)
            print("=" * 50)
            if verdict:
                num_correct += 1
        except:
            continue

    print(f"Accuracy: {num_correct}/{num_total} = {num_correct / num_total:.2%}")


if __name__ == "__main__":
    # main()
    compute_acc()
