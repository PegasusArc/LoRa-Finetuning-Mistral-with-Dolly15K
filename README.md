\documentclass[12pt]{article}

\usepackage[a4paper,margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}

\title{Fine-Tuning Mistral-7B-Instruct with QLoRA on Dolly 15k}
\author{Praneel Sahu}
\date{}

% Code block styling
\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  backgroundcolor=\color{gray!10},
  keywordstyle=\color{blue},
  commentstyle=\color{green!50!black}
}

\begin{document}

\maketitle

\section*{Overview}
This document summarizes the training pipeline for fine-tuning the Mistral-7B-Instruct model using QLoRA on the Databricks Dolly 15k dataset. The implementation uses HuggingFace Transformers, TRL SFTTrainer, PEFT LoRA adapters, and ROUGE evaluation.

\section{Project Structure}
\begin{lstlisting}
train_mistral_lora.py    # Main training script
README.md                # Documentation
requirements.txt         # Dependencies
\end{lstlisting}

\section{Key Features}
\begin{itemize}
    \item Mistral-7B-Instruct as base model
    \item 4-bit QLoRA fine-tuning (bitsandbytes)
    \item TRL SFTTrainer pipeline
    \item Dolly 15k dataset with instruction/response format
    \item ROUGE-1, ROUGE-2, ROUGE-L evaluation
\end{itemize}

\section{Installation}
\begin{lstlisting}[language=bash]
pip install torch transformers datasets accelerate
pip install bitsandbytes peft trl rouge-score
\end{lstlisting}

\section{Training Configuration}
\begin{lstlisting}
Model:            Mistral-7B-Instruct
Quantization:     4-bit NF4
Epochs:           3
LR:               2e-4
Max sequence:     1024
LoRA r:           16
LoRA alpha:       32
LoRA dropout:     0.05
Optimizer:        paged_adamw_32bit
\end{lstlisting}

\section{Running Training}
\begin{lstlisting}[language=bash]
python train_mistral_lora.py
\end{lstlisting}

\section{Evaluation (ROUGE Results)}
\begin{lstlisting}
ROUGE-1: 0.3485
ROUGE-2: 0.1709
ROUGE-L: 0.2724
\end{lstlisting}

These results match expected performance for Dolly 15k fine-tuning.

\section{Example Inference Code}
\begin{lstlisting}[language=Python]
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2")

base = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    load_in_4bit=True
)

model = PeftModel.from_pretrained(base, "./lora_adapter")

messages = [{"role": "user",
             "content": "Why can camels survive long without water?"}]

inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt",
    add_generation_prompt=True).cuda()

output = model.generate(inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
\end{lstlisting}

\section{Optional: Upload to HuggingFace Hub}
\begin{lstlisting}[language=Python]
trainer.push_to_hub("username/mistral-dolly-qlora")
\end{lstlisting}

\end{document}


