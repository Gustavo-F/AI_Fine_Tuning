"""
QLoRA Fine-Tuning for Qwen/Qwen3-4B-Instruct-2507
Adaptado para rodar em 8GB VRAM (RTX 4060).
CSV com colunas: Instructions, Input, Output
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

# ----------------- CONFIGURAÇÕES GERAIS -----------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./out_mistral-7b_lora"
OFFLOAD_DIR = "./offload"
CSV_PATH = "./datasets/dataset_v1.csv"

COL_INSTRUCTION = "Instruction"
COL_INPUT = "Input"
COL_OUTPUT = "Output"

MAX_SEQ_LENGTH = 512
MICRO_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
EPOCHS = 3
LR = 2e-4
# ---------------------------------------------------------

os.makedirs(OFFLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- TOKENIZER --------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------- QUANTIZAÇÃO 4-BIT --------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# -------- MODELO BASE --------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
)

# -------- LORA CONFIG --------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

model.gradient_checkpointing = True

# -------- PROMPT CHATML PARA QWEN --------
def build_prompt(instruction, inp):
    if inp and str(inp).strip():
        return (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}\nInput: {inp}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        return (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

# -------- PREPARAÇÃO DO DATASET --------
ds = load_dataset("csv", data_files=CSV_PATH)["train"]

def process_example(ex):
    instr = ex[COL_INSTRUCTION]
    inp = ex[COL_INPUT]
    out = ex[COL_OUTPUT]

    prompt = build_prompt(instr, inp)
    full_text = prompt + out

    data = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LENGTH)
    prompt_len = len(tokenizer(prompt)["input_ids"])

    labels = [-100] * prompt_len + data["input_ids"][prompt_len:]

    data["labels"] = labels
    return data

tokenized = ds.map(process_example)

# -------- COLLATOR --------
def collator(batch):
    ids = [torch.tensor(b["input_ids"]) for b in batch]
    labels = [torch.tensor(b["labels"]) for b in batch]

    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": (ids != tokenizer.pad_token_id).long()
    }

# -------- TREINAMENTO --------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=collator,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)

print("\nFine-tuning concluído!")
