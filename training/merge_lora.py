import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"    # modelo base
LORA_DIR = "./out_mistral-7b_lora"                 # pasta do seu LoRA
OUTPUT_DIR = "./merged_model"                 # modelo final
torch_dtype = torch.float16

print("Carregando modelo base...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch_dtype,
    device_map="cpu"
)

print("Aplicando LoRA...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)

print("Fazendo merge dos pesos...")
model = model.merge_and_unload()

print("Salvando modelo final...")
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

print("Salvando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✔ Merge concluído! Modelo salvo em:", OUTPUT_DIR)