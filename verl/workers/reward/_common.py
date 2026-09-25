# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path


def decode_sample(tokenizer, batch, prefix="", left_padded=True):
    prompt_ids = batch[f"{prefix}prompts"]
    prompt_length = prompt_ids.shape[-1]
    attention_mask = batch[f"{prefix}attention_mask"]
    valid_prompt_length = attention_mask[:prompt_length].sum()
    response_length = attention_mask[prompt_length:].sum()
    if left_padded:
        prompt_ids = prompt_ids[-valid_prompt_length:] if valid_prompt_length else prompt_ids[:0]
    else:
        prompt_ids = prompt_ids[:valid_prompt_length]
    response_ids = batch[f"{prefix}responses"][:response_length]
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return prompt, response, response_length


def write_reward_logs(samples, rollout_data_dir, global_step):
    log_dir = Path(rollout_data_dir) / "rl_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (log_dir / f"expdata_step_{global_step:05d}.jsonl").open("a", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    except Exception as error:
        print(f"Failed to write logs: {error}")
