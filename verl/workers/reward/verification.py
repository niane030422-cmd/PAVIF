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

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score.verification import (
    instruction_rule,
    instruction_selfverify_compute_score,
    instruction_selfverify_compute_score_val,
)
from ._common import decode_sample, write_reward_logs


__all__ = ["CustomRewardManager"]


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.mode = "val" if compute_score == "instruction_val" else "train"
        if compute_score == "instruction_val":
            self.compute_score = instruction_selfverify_compute_score_val
        elif compute_score == "verification":
            self.compute_score = instruction_selfverify_compute_score
        else:
            self.compute_score = instruction_rule

    def __call__(self, data: DataProto, rollout_data_dir: str, global_step: int) -> tuple:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        accuracy_sum = 0
        constraints = []
        samples = []

        for index in range(len(data)):
            item = data[index]
            prompt, response, response_length = decode_sample(self.tokenizer, item.batch)
            ground_truth = item.non_tensor_batch["extra_info"]["gt"]
            constraints = item.non_tensor_batch["ground_truth"]["instruction_id_list"]

            if self.mode == "train":
                score = self.compute_score(response, ground_truth)
                samples.append({
                    "prompt": prompt,
                    "response": response,
                    "ground_truth": ground_truth,
                    "reward_score": score,
                    "is_validtest": "no",
                })
            else:
                score, accuracy = self.compute_score(response, ground_truth)
                accuracy_sum += accuracy

            if response_length > 0:
                reward_tensor[index, response_length - 1] = score
            if index < self.num_examine:
                print("[prompt]", prompt)
                print("[response]", response)
                print("[reward score]", score)

        if self.mode == "train":
            if getattr(self, "rank", 0) == 0:
                write_reward_logs(samples, rollout_data_dir, global_step)
            return reward_tensor, constraints
        return reward_tensor, accuracy_sum / len(data) if len(data) else 0.0
