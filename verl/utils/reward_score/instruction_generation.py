import math
import re

from ._common import check_instruction, evaluate_rule_constraints


__all__ = ["instruction_selfverify_stage2_compute_score", "instruction_val_compute_score"]

_SCORE_PATTERN = re.compile(r"Score=\\?boxed\{(0(?:\.5)?|1(?:\.0)?)\}")
_BOXED_PATTERN = re.compile(r"\\?boxed\{(0(?:\.5)?|1(?:\.0)?)\}")


def _extract_judge_score(answer):
    match = _SCORE_PATTERN.search(answer or "")
    if match:
        return float(match.group(1))
    tokens = (answer or "").split()
    tail_length = max(1, math.ceil(len(tokens) * 0.15))
    matches = _BOXED_PATTERN.findall(" ".join(tokens[-tail_length:]))
    return float(matches[-1]) if matches else None


def instruction_val_compute_score(answer, item):
    results = [
        check_instruction(answer, instruction_id, kwargs)
        for instruction_id, kwargs in zip(item["instruction_id_list"], item["kwargs"])
    ]
    passed, total = sum(results), len(results)
    score = passed / total if total else 0.0
    return int(score == 1), passed, total, score


def instruction_selfverify_stage2_compute_score(answer, initial_ans, item):
    direct_score, passed, checked = evaluate_rule_constraints(initial_ans, item)
    if direct_score is not None:
        return direct_score
    score = _extract_judge_score(answer)
    if score is None:
        return 0.0
    count = len(item["constraints"])
    return (score * count + passed) / (count + checked) if count + checked else 0.0
