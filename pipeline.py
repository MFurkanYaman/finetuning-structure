from peft import PeftModel, PeftConfig
from transformers import pipeline,AutoTokenizer,AutoModelForCausalLM
from huggingface_hub import login

import torch

login("your token")

print("Login token girildi.")

model_path="./resultsGala/checkpoint-386"
base_model="tolgadev/llama-2-7b-ruyallm"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# LoRA adaptörünü yükleme
config = PeftConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)
# LoRA ağırlıklarını yükleme
model = PeftModel.from_pretrained(model, model_path)


# Inference için fonksiyon
def generate_response(prompt, max_length=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.1,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Test etme
prompt = ("""Galatasaray Spor Kulübü hakkında bana bilgi verir misin ?""")

response = generate_response(prompt)
print(response)


