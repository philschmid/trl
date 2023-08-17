# Rank Responses to Align Language Models with Human Feedback (RRHF)

Implementation:
1️⃣ Sample responses from various sources, including the model, other LLMs like GPT, human-labeled data, and even low-quality data (needed for learning to rank)
2️⃣ Rank/score those responses using humans or Reward models
3️⃣ Train model using RRHF (Ranking loss + Cross entropy)

## Setup

`pip install -r requirements.txt`

## Example 

The `rrhf` folder contains an example implementation of the [RRHF: Rank Responses to Align Language Models with Human Feedback without tears
](https://arxiv.org/abs/2304.05302) using the new `RRHFTrainer` of `trl`. The example can be used to reproduce the results of the paper.

### 1. Generate samples 

First step is to generate samples from various sources. Since we are going to use the `HH` dataset this step is not needed. 

### 2. Rank samples


