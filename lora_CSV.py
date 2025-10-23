from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
import json

# تحميل البيانات
with open("cleaned_texts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [{"text": sample["text"]} for sample in data]
dataset = Dataset.from_list(texts)

# تحميل Tokenizer و Model عربي مخصص
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/aragpt2-base")
tokenizer.pad_token = tokenizer.eos_token  # تعيين pad_token

model = AutoModelForCausalLM.from_pretrained("aubmindlab/aragpt2-base")

# دالة الترميز
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=100,

    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# إعداد LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# إعدادات التدريب
training_args = TrainingArguments(
    output_dir="./aragpt2_lora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=300,#100#200
    learning_rate=2e-4,
    fp16=False,
    logging_dir="./logs",
    save_strategy="epoch",
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

# مدرب
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# بدء التدريب
trainer.train()

# حفظ النموذج
model.save_pretrained("./aragpt2_lora_textgen")
tokenizer.save_pretrained("./aragpt2_lora_textgen")
