# 0. imports
from dataclasses import dataclass, field
from random import randrange
from typing import Dict, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from torch.utils.data import Dataset

from trl import RRHFTrainer, RRHFDataCollatorWithPadding, RRHFTrainingArguments


"""
RRHF example:
```bash
python examples/research_projects/rrhf/train.py \
  --model_name_or_path gpt2 \
  --dataset_id philschmid/hh-rrhf-dahoas-gptj-rm-25k \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --bf16 \
  --output_dir ./tmp/rrhf
"""


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # training parameters
    model_name_or_path: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    dataset_id: Optional[str] = field(default="xsum", metadata={"help": "the dataset id"})


def main():
    parser = HfArgumentParser([ScriptArguments, RRHFTrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. load the dataset and create collator
    dataset = load_dataset(script_args.dataset_id, split="train")
    # dataset = ScoreDataset(dataset)
    data_collator = RRHFDataCollatorWithPadding(tokenizer=tokenizer)
    # print random sample
    print(dataset[randrange(len(dataset))])

    # 3. initialize the RRHF trainer
    trainer = RRHFTrainer(
        model=model, tokenizer=tokenizer, args=training_args, train_dataset=dataset, data_collator=data_collator
    )

    # 4. start training
    trainer.train()

    # 5. save the trained model
    trainer.save_model()


if __name__ == "__main__":
    main()
