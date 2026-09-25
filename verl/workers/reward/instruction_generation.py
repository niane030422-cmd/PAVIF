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
from ...utils.reward_score.instruction_generation import (
    instruction_selfverify_stage2_compute_score,
    instruction_val_compute_score,
)
from ._common import decode_sample, write_reward_logs


__all__ = ["self_CustomRewardManager"]


class self_CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        if compute_score == "instruction":
            self.compute_score = instruction_selfverify_stage2_compute_score
            self.mode = "train"
        elif compute_score == "instruction_val":
            self.compute_score = instruction_val_compute_score
            self.mode = "val"
        else:
            raise NotImplementedError(f"Unsupported reward: {compute_score}")

    def __call__(self, data: DataProto, rollout_data_dir: str, global_step: int) -> tuple:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        passed_constraints = total_constraints = passed_samples = 0
        response_info = []
        samples = []

        for index in range(len(data)):
            item = data[index]
            prompt, response, response_length = decode_sample(self.tokenizer, item.batch)
            response = response.replace("\x08oxed", r"\boxed")
            ground_truth = item.non_tensor_batch["ground_truth"]
            constraints = ground_truth["instruction_id_list"]
            verification_prompt = verification_response = None

            if self.mode == "train":
                verification_ids = item.batch.get("verification_responses")
                if verification_ids is not None and verification_ids.sum() > 0:
                    verification_prompt, verification_response, _ = decode_sample(
                        self.tokenizer, item.batch, prefix="verification_", left_padded=False
                    )
                    verification_response = verification_response.replace("\x08oxed", r"\boxed")
                score = self.compute_score(verification_response, response, ground_truth)
                response_info.append({"cc": constraints, "rl": response_length})
                samples.append({
                    "prompt": prompt,
                    "response": response,
                    "verification_prompt": verification_prompt,
                    "verification_response": verification_response,
                    "ground_truth": ground_truth,
                    "reward_score": score,
                    "is_validtest": "no",
                    "constraints": constraints,
                })
            else:
                all_passed, passed, total, score = self.compute_score(response, ground_truth)
                passed_constraints += passed
                total_constraints += total
                passed_samples += all_passed

            if response_length > 0:
                reward_tensor[index, response_length - 1] = score
            if index < self.num_examine:
                print("【prompt】", prompt)
                print("【response】", response)
                if self.mode == "train":
                    print("【verification_prompt】", verification_prompt)
                    print("【verification_response】", verification_response)
                print("【ground_truth】", ground_truth)
                print("【reward score】", score)

        if self.mode == "train":
            if getattr(self, "rank", 0) == 0:
                write_reward_logs(samples, rollout_data_dir, global_step)
            return reward_tensor, response_info
        score_c = passed_constraints / total_constraints if total_constraints else 0.0
        score_l = passed_samples / len(data) if len(data) else 0.0
        return reward_tensor, score_c, score_l
