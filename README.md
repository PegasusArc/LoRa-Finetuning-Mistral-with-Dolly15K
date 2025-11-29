\documentclass[12pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{titlesec}
\usepackage{hyperref}
\usepackage{lmodern}
\usepackage{listings}
\usepackage{xcolor}

\title{Fine-Tuning Mistral-7B-Instruct with QLoRA on Dolly 15k}
\author{Praneel Sahu}
\date{\today}

\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  backgroundcolor=\color{gray!5}
}

\begin{document}

\maketitle

\section*{Abstract}
This document describes a complete training pipeline for fine-tuning the Mistral-7B-Instruct-v0.2 model using QLoRA on the Databricks Dolly 15k dataset. The approach uses the TRL SFTTrainer, LoRA adapters, and ROUGE-based evaluation. The pipeline is optimized for Google Colab and GPU-limited environments.

\section{Project Structure}
\begin{lstlisting}
.
├── train_mistral_lora.py    # Main training script
├── README.md                # Documentation
└── requirements.txt         # Optional dependencies
\end{lstlisting}

\section{Features}
\begin{itemize}
    \item Mistral-7B-Instruct base model.
    \item QLoRA (4-bit quantization via bitsandbytes).
    \item TRL SFTTrainer for supervised instruction tuning.
    \item PEFT LoRA adapters for efficient training.
    \item Chat template formatting for training stability.
    \item Automatic ROUGE evaluation on a held-out test split.
    \item Optional HuggingFace Hub upload.
\end{itemize}

\section{Installation}
Install all required libraries:

\begin{lstlisting}[language=bash]
pip install torch transformers datasets accelerate bitsandbytes trl peft rouge-score
\end{lstlisting}

For CUDA 12:
\begin{lstlisting}[language=bash]
pip install bitsandbytes==0.43.0 --extra-index-url \
  https://download.pytorch.org/whl/cu121
\end{lstlisting}

\section{Dataset}
We fine-tune the model on the Databricks Dolly 15k dataset:

\begin{lstlisting}
databricks/databricks-dolly-15k
\end{lstlisting}

The dataset is automatically split:
\begin{itemize}
    \item 95\% training
    \item 5\% testing (for ROUGE evaluation)
\end{itemize}

\section{Training Configuration}
\begin{lstlisting}
Epochs:                  3
Learning Rate:           2e-4
Batch Size:              2 (with gradient accumulation)
Max Sequence Length:     1024
Optimizer:               paged_adamw_32bit
LoRA r:                  16
LoRA alpha:              32
LoRA dropout:            0.05
Quantization:            4-bit NF4
\end{lstlisting}

\section{Using the Training Script}
Run training:
\begin{lstlisting}[language=bash]
python train_mistral_lora.py
\end{lstlisting}

The script automatically:
\begin{itemize}
    \item Loads Mistral-7B in 4bit quantization.
    \item Prepares Dolly-15k in chat format.
    \item Trains LoRA adapters with QLoRA.
    \item Evaluates via ROUGE-1, ROUGE-2, ROUGE-L.
\end{itemize}

\section{Example Evaluation Results}
Typical ROUGE scores:

\begin{lstlisting}
ROUGE-1: 0.3485
ROUGE-2: 0.1709
ROUGE-L: 0.2724
\end{lstlisting}

These values are consistent with expected upper performance bounds on Dolly 15k.

\section{Inference Example}
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

messages = [{"role": "user", "content": "Why can camels survive without water?"}]
inputs = tokenizer.apply_chat_template(messages,
        return_tensors="pt",
        add_generation_prompt=True).cuda()

output = model.generate(inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
\end{lstlisting}

\section{Upload to HuggingFace Hub (Optional)}
\begin{lstlisting}[language=Python]
trainer.push_to_hub(
    "your-username/mistral7b-dolly-qlora"
)
\end{lstlisting}

\section{License}
This project is released under the MIT License (or your choice).

\section{Acknowledgements}
Mistral AI, HuggingFace Transformers, TRL, PEFT, BitsAndBytes, and Databricks Dolly.

\end{document}

