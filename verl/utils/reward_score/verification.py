import re

from ._common import evaluate_rule_constraints


__all__ = [
    "instruction_rule",
    "instruction_selfverify_compute_score",
    "instruction_selfverify_compute_score_val",
]

_SCORE_PATTERN = re.compile(r"Score=\s*\$?_?\\?boxed\{(0(?:\.5)?|1(?:\.0)?)\}")


def _extract_judge_score(answer):
    match = _SCORE_PATTERN.search(answer or "")
    return float(match.group(1)) if match else None


def _discretize_score(score):
    if score == 1:
        return 1
    return 0.5 if score >= 0.5 else 0


def instruction_selfverify_compute_score(answer, item):
    score = _extract_judge_score(answer)
    return 1 - abs(score - float(item[0])) if score is not None else 0.0


def instruction_selfverify_compute_score_val(answer, item):
    score = _extract_judge_score(answer)
    if score is None:
        return 0.0, 0
    target = float(item[0])
    return 1 - abs(score - target), int(_discretize_score(score) == _discretize_score(target))


def instruction_rule(initial_ans, item):
    direct_score, passed, _ = evaluate_rule_constraints(initial_ans, item)
    if direct_score is not None:
        return direct_score
    count = len(item["constraints"])
    return passed * (1 / count) if count else 0.0
