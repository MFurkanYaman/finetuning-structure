import time
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

# Hugging Face token ile giriş
login("your token")  

print("Login token girildi.")

# Model ve veri yolu yapılandırmaları
base_model = "tolgadev/llama-2-7b-ruyallm"
wiki_data = "/home/fyaman/Desktop/llama2/dataset/galatasaray.csv"  # CSV dosya
new_model = "llama-2-7b-myFineTuning"

# Veri setini yükleme
dataset = load_dataset('csv', data_files=wiki_data)

print("Dataset yüklendi")

# 4-bit sayısallaştırma yapılandırması
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=True,
    llm_int8_threshold=6.,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
)
print("quant config yapıldı.")

# Model yükleme
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}  # GPU kullan
)
model.config.use_cache = False
model.config.pretraining_tp = 1

print("model yüklendi")

# Tokenizer yükleme
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Tokenizer yüklendi")

# PEFT parametreleri
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
print("Peft params yüklendi")

# Eğitim parametreleri
training_params = TrainingArguments(
    output_dir="./resultsGala",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

print("Eğitim parametreleri alındı.")

time.sleep(3)

# Eğitici oluşturma ve başlatma
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    peft_config=peft_params,
    tokenizer=tokenizer,
    args=training_params,
)

print("Eğitim başladı")
# Eğitimi başlat
trainer.train()
print("eğitim bitti")