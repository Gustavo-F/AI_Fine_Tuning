import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

# -------- CONFIGURAÇÕES --------
BASE_MODEL = os.getenv("MODEL")
LORA_DIR = os.getenv("OUTPUT_DIR")
MAX_NEW_TOKENS = 300

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------

print("Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configuração para carregar em 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

print("Carregando modelo base em 4-bit...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Carregar adaptador LoRA
print("Carregando LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model = model.to(DEVICE)

def build_prompt(instruction, inp=None):
    """
    Template recomendado para Qwen 3 Instruct.
    Extremamente simples (Qwen já entende formato user/assistant).
    """
    if inp:
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

def generate_response(instruction, inp=None):
    prompt = build_prompt(instruction, inp)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.cuda.amp.autocast():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Exemplo de teste
if __name__ == "__main__":
    input_a = "logMessage --message \"\r\nYour address is: ${street} ${number} - ${district} - ${city}\""
    resposta = generate_response(
        f"Reescreva o script do IBM RPA a seguir em python: {input_a}",
    )
    print("\n----- RESPOSTA -----\n")
    print(resposta)