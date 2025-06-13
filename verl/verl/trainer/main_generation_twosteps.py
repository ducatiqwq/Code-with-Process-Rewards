import os
import csv
import json
import re
import numpy as np
import pandas as pd
import ray
import hydra
from omegaconf import OmegaConf
from pprint import pprint
from tabulate import tabulate

from transformers import AutoTokenizer
from verl.utils.model import compute_position_id_with_mask
from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs, makedirs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


def get_prompt_stage1(input_prompt):
    problem = re.search(r'passes all tests.\n\n(.*?)\n\nSample Input 1:', input_prompt, re.DOTALL)
    if problem:
        problem = problem.group(1).strip()
    else:
        problem = re.search(r'### Problem Description:\n(.*?)\n###', input_prompt, re.DOTALL)
        if problem:
            problem = problem.group(1).strip()
        else:
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

def get_prompt_stage2(input_prompt, stage1_output):
    problem = re.search(r'passes all tests.\n\n(.*?)\n\nSample Input 1:', input_prompt, re.DOTALL)
    if problem:
        problem = problem.group(1).strip()
    else:
        raise ValueError("Could not find problem description in the input prompt.")
    incomplete_code = re.search(r'```python(.*?)```', stage1_output, re.DOTALL)
    if incomplete_code:
        incomplete_code = incomplete_code.group(1).strip()
    else:
        raise ValueError("Could not find incomplete code in the stage 1 output.")
    
    prompt_s2 = f"### Task\nYou are an expert Python programmer. You will be given a question (problem specification) and an incomplete code snippet that contains the overall pipeline (in the main block) and masked sub-tasks (functions with docstrings but no implementation).\n\nYour task is to implement all masked sub-tasks (possibly through adding new imports, classes, functions) in the specified order, and the main block must be left unmodified. The generated code must be a correct Python program that matches the specification and passes all tests.\n\n###{problem}\n### Format\nRead the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n### Incomplete Code Snippet:\n{incomplete_code}\n### Answer: (use the provided format with backticks)"
    return prompt_s2


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    pprint(OmegaConf.to_container(config, resolve=True))

    # --- Load or generate the DataFrame ---
    if os.path.exists(config.data.output_path):
        try:
            dataset = pd.read_parquet(config.data.output_path)
        except:
            # fallback JSON / Polars
            alt = config.data.output_path.replace(".parquet", ".json")
            try:
                dataset = pd.read_json(alt)
            except:
                import polars as pl
                dataset = pl.read_parquet(config.data.output_path).to_pandas()
    else:
        # copy model1 locally for tokenizer
        local_model1 = copy_local_path_from_hdfs(config.model.stage1.path)
        tokenizer = AutoTokenizer.from_pretrained(local_model1, use_fast=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # read raw prompts
        try:
            df = pd.read_parquet(config.data.path)
        except:
            df = pd.read_json(config.data.path.replace(".parquet", ".json"))
        chat_lst = df[config.data.prompt_key].tolist()

        # set up Ray for *both* stages
        ray.init(ignore_reinit_error=True)

        def make_wg(global_cfg, model_path: str, role_name: str):
            role_gpu_map = {"stage1": 0, "stage2": 1}
            # clone + tweak config
            stage_cfg = global_cfg.copy()
            stage_cfg.model.path = model_path


            ray_cls = RayClassWithInitArgs(
                cls=ActorRolloutRefWorker,            # <-- only this handle is GPU-aware
                config=stage_cfg,
                role=role_name,
            )

            pool = RayResourcePool(process_on_nodes=[1] * global_cfg.trainer.nnodes)
            wg = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=ray_cls)
            wg.init_model()
            return wg

        # Stage 1 worker group
        wg1 = make_wg(config, config.model.stage1.path, "rollout")
        # Stage 2 worker group
        wg2 = make_wg(config, config.model.stage2.path, "rollout")

        output_rows = []
        batch_size = config.data.batch_size
        dp_size = wg1.world_size  # assume both groups have same world_size
        num_batches = (len(chat_lst) + batch_size - 1) // batch_size

        for i in range(num_batches):
            batch_prompts = chat_lst[i*batch_size : (i+1)*batch_size]

            # --- Stage 1: run model1 ---
            # build stage-1 prompts
            s1_inp = [get_prompt_stage1(p) for p in batch_prompts]

            # tokenize and shard (exactly as your original code did)
            enc = tokenizer.apply_chat_template(
                s1_inp,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors="pt",
                return_dict=True,
                tokenize=True
            )
            data1 = DataProto.from_dict({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "position_ids": compute_position_id_with_mask(enc["attention_mask"])
            })
            # pad to dp_size
            real_bs = data1.batch["input_ids"].shape[0]
            if real_bs % dp_size != 0:
                dummy = data1[: dp_size - (real_bs % dp_size)]
                data1 = DataProto.concat([data1, dummy])

            out1 = wg1.generate_sequences(data1)[:real_bs]
            texts1 = tokenizer.batch_decode(
                out1.batch["input_ids"][:, -config.rollout.response_length :],
                skip_special_tokens=False
            )
            texts1 = [t.replace(tokenizer.pad_token, "") for t in texts1]

            # --- Stage 2: run model2 on each Stage-1 output ---
            s2_inp = [get_prompt_stage2(orig, st1) for orig, st1 in zip(batch_prompts, texts1)]

            enc2 = tokenizer.apply_chat_template(
                s2_inp,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors="pt",
                return_dict=True,
                tokenize=True
            )
            data2 = DataProto.from_dict({
                "input_ids": enc2["input_ids"],
                "attention_mask": enc2["attention_mask"],
                "position_ids": compute_position_id_with_mask(enc2["attention_mask"])
            })
            real_bs2 = data2.batch["input_ids"].shape[0]
            if real_bs2 % dp_size != 0:
                dummy2 = data2[: dp_size - (real_bs2 % dp_size)]
                data2 = DataProto.concat([data2, dummy2])

            out2 = wg2.generate_sequences(data2)[:real_bs2]
            texts2 = tokenizer.batch_decode(
                out2.batch["input_ids"][:, -config.rollout.response_length :],
                skip_special_tokens=False
            )
            texts2 = [t.replace(tokenizer.pad_token, "") for t in texts2]

            output_rows.extend(texts2)

        # reshape and attach as before
        ndata = len(output_rows) // config.data.n_samples
        responses = np.array(output_rows).reshape(ndata, config.data.n_samples).tolist()
        df["responses"] = responses
        makedirs(os.path.dirname(config.data.output_path), exist_ok=True)
        df.to_parquet(config.data.output_path)
        dataset = df  # for the next part

    # --- evaluation (unchanged) ---
    prompts = dataset[config.data.prompt_key]
    responses = dataset["responses"]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    total_scores = []
    for src, p, resp_list, rd in zip(data_sources, prompts, responses, reward_model_data):
        reward_fn = select_reward_fn(src)
        score_list = []
        for r in resp_list:
            try:
                score_list.append(reward_fn(r, rd["ground_truth"]))
            except:
                score_list.append(reward_fn(src, r, rd["ground_truth"]))
        total_scores.append(score_list)
        if max(score_list) == 1:
            passes += 1

    pass_at_n = passes / len(dataset)
    pass_at_1 = np.mean([[s[0] for s in total_scores]])

    # append to CSV
    out_dir = os.path.dirname(config.data.output_path)
    stats = {
        "model1": config.model.stage1.path,
        "model2": config.model.stage2.path,
        "pass@1": pass_at_1,
        f"pass@{config.data.n_samples}": pass_at_n
    }
    csv_path = os.path.join(out_dir, "pass.csv")
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=stats.keys())
        if f.tell() == 0:
            w.writeheader()
        w.writerow(stats)

    # print table
    print(tabulate([[k, v] for k, v in stats.items()], headers=["Metric", "Value"], tablefmt="grid"))

    # write per-sample scores
    json.dump(total_scores, open(os.path.join(out_dir, "results.json"), "w"))


def select_reward_fn(data_source):
    if data_source == "lighteval/MATH":
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        from rllm.rewards.rl_reward import rllm_reward_fn
        return rllm_reward_fn


if __name__ == "__main__":
    main()
