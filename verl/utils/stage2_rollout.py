import os
import re
from pathlib import Path

_DEFAULT_PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts"
_PROMPT_ROOT = Path(os.environ.get("PROMPT_ROOT", _DEFAULT_PROMPT_ROOT))


def _load_template(filename: str) -> str:
    return (_PROMPT_ROOT / filename).read_text(encoding="utf-8")


cot_verify_prompt_template_en = _load_template("cot_verify_en.txt")
cot_verify_prompt_template_zh = _load_template("cot_verify_zh.txt")


def max_length_substr(pattern, s):
    substrs = re.findall(pattern, s)
    max_length = max((len(substr.strip()) for substr in substrs), default=0)
    total_lenght = sum(len(s.strip()) for s in substrs)
    return max_length, total_lenght

def check_zh_en_mix(s):
    chinese_pattern = r'[\u4e00-\u9fff]+'
    english_pattern = r'[A-Za-z]+'
    
    has_chinese = re.search(chinese_pattern, s) is not None
    has_english = re.search(english_pattern, s) is not None
    
    max_chinese_length = 0
    max_english_length = 0
    total_chinese_lenght = 0
    total_english_lenght = 0
    
    if has_chinese:
        max_chinese_length, total_chinese_lenght = max_length_substr(chinese_pattern, s)
    
    if has_english:
        max_english_length, total_english_lenght = max_length_substr(english_pattern, s)
    
    if has_chinese and has_english:
        string_type = "Mixed"
    elif has_chinese:
        string_type = "Chinese"
    elif has_english:
        string_type = "English"
    else:
        string_type = "Other"
        
    return string_type, max_chinese_length, max_english_length, total_chinese_lenght, total_english_lenght

def check_zh_en_ratio(s):
    cls, max_zh, max_en, total_zh, totel_en = check_zh_en_mix(s)
    zhratio = total_zh / (totel_en + total_zh + 1)
    enratio = totel_en / (totel_en + total_zh + 1)
    return zhratio, enratio

def is_chinese(origin_prompt):
    zhratio, enratio = check_zh_en_ratio(origin_prompt)
    return zhratio > 0.5

def apply_prompt_template(question, response):
    if is_chinese(question):
        return cot_verify_prompt_template_zh.replace("{question}", question).replace("{answer}", response)
    else:
        return cot_verify_prompt_template_en.replace("{question}", question).replace("{answer}", response)

''''''
