#!/bin/bash
#SBATCH --job-name=prompt_baseline_a
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home5/s6398820/projects/cs336/assignment5/outputs/logs/section3/prompt_baseline_a_%j.out
#SBATCH --error=/home5/s6398820/projects/cs336/assignment5/outputs/logs/section3/prompt_baseline_a_%j.err

cd /home5/s6398820/projects/cs336/assignment5

mkdir -p outputs/logs/section3

source .venv/bin/activate
python scripts/section3/prompting_baselines.py cs336_alignment/prompts/r1_zero.prompt data/gsm8k/test.jsonl outputs/section3 allenai/OLMo-2-0425-1B
python scripts/section3/prompting_baselines.py cs336_alignment/prompts/question_only.prompt data/gsm8k/test.jsonl outputs/section3 allenai/OLMo-2-0425-1B
python scripts/section3/prompting_baselines.py cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt data/gsm8k/test.jsonl outputs/section3 allenai/OLMo-2-0425-1B

