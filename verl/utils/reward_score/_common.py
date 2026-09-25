import inspect
import re

from mathruler.grader import extract_boxed_content
from modules import instructions_registry


SOFT_INSTRUCTIONS = {
    "soft_content:language",
    "soft",
    "soft_content:open_ended",
    "situation:suggestion",
    "situation:role_play",
    "situation:story_generation",
    "style:open_ended",
}


def score_boxed_answer(answer, expected):
    answer = (answer or "").replace("\x08", "b").replace("\\b", "b")
    last_answer = None
    index = 0
    while index < len(answer):
        start = answer.find("oxed{", index)
        if start < 0:
            break
        start += len("oxed{")
        depth = 1
        index = start
        while index < len(answer) and depth:
            if answer[index] == "{":
                depth += 1
            elif answer[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            last_answer = answer[start:index - 1].strip()

    if last_answer is None:
        return 0.0
    if last_answer == "self-contradiction":
        last_answer = "self_contradiction"
    return float(last_answer.replace("\\", "").strip() == str(expected).replace("\\", "").strip())


def check_instruction(answer, instruction_id, kwargs):
    instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    parameters = inspect.signature(instruction.build_description).parameters
    instruction.build_description(**{key: value for key, value in kwargs.items() if key in parameters})
    return bool(answer and answer.strip() and instruction.check_following(answer))


def evaluate_rule_constraints(answer, item):
    passed = checked = 0
    constraints = item["constraints"]
    for instruction_id, kwargs, _ in zip(item["instruction_id_list"], item["kwargs"], constraints):
        if instruction_id == "light":
            return score_boxed_answer(answer, constraints[0]), passed, checked
        if instruction_id == "light_choice":
            boxed_answer = extract_boxed_content(answer or "")
            if boxed_answer == "None":
                matches = re.findall(r"boxed{(.*?)}", answer or "")
                if matches:
                    boxed_answer = matches[-1]
            return float(boxed_answer == constraints[0]), passed, checked
        if instruction_id not in SOFT_INSTRUCTIONS:
            passed += check_instruction(answer, instruction_id, kwargs)
            checked += 1
    return None, passed, checked
