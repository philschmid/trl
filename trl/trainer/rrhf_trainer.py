# RRHF Authors: Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, Fei Huang, 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass, field
import warnings
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments


@dataclass
class RRHFTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch")
    rrhf_weight: float = field(default=100.0)
    length_penalty: float = field(default=1.0)
    remove_unused_columns: bool = field(
        default=False
    )  # This is set to avoid having the trainer remove our non input_id columns


class RRHFTrainer(Trainer):
    def gather_logits_labels(self, logits, labels):
        mask = (labels != -100).float()
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == -100] = 0
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        output = output * mask  # B * L
        return output

    def get_score(self, logit_label, labels):
        mask = (labels != -100).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length**self.args.length_penalty)
        return scores

    def rrhf_loss(self, scores, rw_scores):
        diff = scores.unsqueeze(0) - scores.unsqueeze(-1)  # b * b
        rw_diff = rw_scores.unsqueeze(0) - rw_scores.unsqueeze(-1)  # b * b
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)[0]
        return -diff[aval].sum()

    def sft_loss(self, logit_label, rw_scores):
        max_idx = torch.argmax(rw_scores)
        return -logit_label[max_idx].mean()

    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[
            0
        ]  # (batch * cand) * L * V
        logits = F.log_softmax(logits, dim=-1)
        logit_label = self.gather_logits_labels(logits, inputs["labels"])
        scores = self.get_score(logit_label, inputs["labels"])
        rrhf_loss = self.rrhf_loss(scores, inputs["scores"])
        sft_loss = self.sft_loss(logit_label, inputs["scores"])
        loss = self.args.rrhf_weight * rrhf_loss + sft_loss
        return (loss, scores) if return_outputs else loss
