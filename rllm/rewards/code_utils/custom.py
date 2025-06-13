import re
import logging
from typing import List, Tuple

from .livecodebench import convert_line_to_decimals, get_stripped_lines
from ..reward_types import RewardConfig


def postprocess_code(sample, code: str) -> Tuple[str, List[str]]:
    main_with_debug = sample["codes"]["main_with_debug"]
    tasks = re.findall(r'print\s*\(\s*[\'"]<([^/]*?)>[\'"]\s*\)', main_with_debug)

    # Ensure the tasks are implemented in the same order as in the main_with_debug
    last_pos = -1
    for task in tasks:
        match = re.search(rf"def\s*{re.escape(task)}", code)
        if not match:
            return "", []

        pos = match.start()
        if pos < last_pos:
            return "", []
        last_pos = pos

    for task in tasks:
        pattern_begin = rf'print\s*\(\s*[\'"]<{re.escape(task)}>[\'"]\s*\)'
        pattern_end = rf'print\s*\(\s*[\'"]</{re.escape(task)}>[\'"]\s*\)'

        begin_match = re.search(pattern_begin, main_with_debug)
        end_match = re.search(pattern_end, main_with_debug)

        if begin_match and end_match:
            # Extract the code strictly between the 'print' markers
            start_pos = begin_match.start()
            end_pos = end_match.end()
            task_all_code = main_with_debug[start_pos: end_pos]
            task_call_code = re.sub(r'print\s*\([^)]*\)', '', task_all_code).strip()

            if task_call_code in code:
                last_match_pos = code.rfind(task_call_code)
                code = code[:last_match_pos] + task_all_code + code[last_match_pos + len(task_call_code):]
            else:
                # logging.error(f"Task {task} call not found in the code.")
                return "", []
        else:
            # logging.error(f"The dataset is corrupted. Missing markers for task: {task}")
            return "", []

    return code.strip(), tasks


def compare_results(prediction: str, gt_out: str) -> bool:
    stripped_prediction_lines = get_stripped_lines(prediction)
    stripped_gt_out_lines = get_stripped_lines(gt_out)

    if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
        return False

    for output_line_idx, (
        stripped_prediction_line,
        stripped_gt_out_line,
    ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
        ## CASE 1: exact match
        if stripped_prediction_line == stripped_gt_out_line:
            continue

        ## CASE 2: element-wise comparision
        ## if there are floating elements
        ## use `decimal` library for good floating point comparision
        ## otherwise gotcha: np.isclose(50000000000000000, 50000000000000001) = True
        ## note that we should always be able to convert to decimals

        success, decimal_prediction_line = convert_line_to_decimals(
            stripped_prediction_line
        )
        if not success:
            return False

        success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
        if not success:
            return False

        if decimal_prediction_line != decimal_gtout_line:
            return False

    return True


def get_task_rewards(output: str, gt_output: str, tasks: List[str], config: RewardConfig) -> List[float]:
    rewards = []
    for idx, task in enumerate(tasks):
        def try_match(output: str):
            pattern_begin = f"<{task}>\n"
            pattern_end = f"</{task}>\n"
            if output.count(pattern_begin) != 1 or output.count(pattern_end) != 1:
                raise ValueError(f"Output does not contain exactly one <{task}> and </{task}> tag.")

            begin_match = re.search(pattern_begin, output).end()    # type: ignore
            end_match = re.search(pattern_end, output).start()      # type: ignore
            return output[begin_match:end_match].strip()
        
        if idx != len(tasks) - 1:
            rewards.append(0) # baseline
            continue

        try:
            output_task = try_match(output)
            gt_output_task = try_match(gt_output)
            if not compare_results(output_task, gt_output_task):
                raise ValueError(f"Output for task {task} does not match ground truth.")
            rewards.append(config.step_correct_reward)
        except ValueError as e:
            # if idx == len(tasks) - 1:
                # If it's the last task, we give a final incorrect reward
            rewards.append(config.final_incorrect_reward)  
            # else:
                # rewards.append(config.soft_step_incorrect_reward)
            break

    # step-aware GAE
    last_ga = 0
    for i in reversed(range(len(rewards))):
        rewards[i] += last_ga * config.step_aware_gae_lambda
        last_ga = rewards[i]

    return rewards


def get_token_rewards(response_ids, tasks: List[str], task_level_rewards: List[float], tokenizer) -> List[float]:
    last_pos = 0
    response = tokenizer.decode(response_ids)

    def get_task_last_token(task: str) -> int:
        code_blocks = re.finditer(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
        code_blocks = reversed(list(code_blocks))
        code_start, code_end = next(code_blocks).span()

        task_start_match = re.search(rf"\ndef\s*{re.escape(task)}", response[code_start: code_end])
        assert task_start_match is not None

        task_end_match = re.search(r"\n\S", response[code_start + task_start_match.end(): code_end])
        assert task_end_match is not None

        # task_start = code_start + task_start_match.start()
        task_end = code_start + task_start_match.end() + task_end_match.start()
        num_tokens = tokenizer.encode(response[:task_end])
        return len(num_tokens)

    rewards = [0.0] * len(response_ids)
    for task, reward in zip(tasks, task_level_rewards):
        cur_pos = get_task_last_token(task)
        for i in range(last_pos, cur_pos):
            rewards[i] = reward
        last_pos = cur_pos

    return rewards


if __name__ == "__main__":
    sample = {
        "codes": {
            "main_with_debug": "if __name__ == \"__main__\":\n    print(\"<read_input>\")\n    n, a = read_input()\n    print(n, a)\n    print(\"</read_input>\")\n\n    print(\"<solve>\")\n    result = solve(n, a)\n    print(result)\n    print(\"</solve>\")\n\n    print(\"<print_result>\")\n    print_result(result)\n    print(\"</print_result>\")"
        }
    }
    code = "def read_input():\n    \"\"\"Read input from standard input.\"\"\"\n    n = int(input())\n    a = list(map(int, input().split()))\n    return n, a\n\ndef solve(n, a):\n    \"\"\"\n    Solve the problem by finding the second largest element's index.\n    \n    Args:\n        n: The length of the array\n        a: The array of integers\n    \n    Returns:\n        int: The 1-indexed position of the second largest element\n    \"\"\"\n    largest = -float('inf')\n    second_largest = -float('inf')\n    largest_idx = -1\n    second_largest_idx = -1\n    \n    for i in range(n):\n        if a[i] > largest:\n            second_largest = largest\n            second_largest_idx = largest_idx\n            largest = a[i]\n            largest_idx = i\n        elif a[i] > second_largest and a[i] < largest:\n            second_largest = a[i]\n            second_largest_idx = i\n    \n    # Return 1-indexed position\n    return second_largest_idx + 1\n\ndef print_result(result):\n    \"\"\"Print the result.\"\"\"\n    print(result)\n\nif __name__ == \"__main__\":\n    n, a = read_input()\n    result = solve(n, a)\n    print_result(result)"

    import transformers
    model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    response = "This is the code:```python\n" + code + "\n```"
    tokenized_response = tokenizer.encode(response)

    rewards = get_token_rewards(
        tokenized_response,
        ["read_input", "solve", "print_result"],
        [1.0, -1.0, 0.0],
        tokenizer
    )
    print(rewards)