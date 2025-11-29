Fine-Tuning Mistral-7B-Instruct with QLoRA on Dolly 15k

This repository contains a full training pipeline for fine-tuning Mistral-7B-Instruct-v0.2 using QLoRA on the Databricks Dolly 15k instruction dataset.
The training script implements efficient PEFT fine-tuning, dataset preprocessing, training with SFTTrainer, and ROUGE-based evaluation.


Features

Mistral-7B-Instruct base model

QLoRA (4-bit NF4 quantization) using bitsandbytes

TRL’s SFTTrainer for supervised fine-tuning

LoRA adapters for lightweight training

Chat-template formatting for proper dialog structure

Automatic ROUGE evaluation on a 5% held-out test split

Optional:

Push trained LoRA adapters to HuggingFace Hub

Merge LoRA with base model
