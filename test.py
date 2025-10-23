

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

def generate_long_text(prompt, chunks=3, chunk_size=400):
    """
    توليد نص طويل بتقسيمه إلى عدة أجزاء باستخدام حلقة for
    prompt     : نص البداية
    chunks     : عدد الأجزاء التي نريد توليدها
    chunk_size : عدد التوكنز لكل جزء
    """
    tokenizer = AutoTokenizer.from_pretrained("./aragpt2_lora_textgen")
    model = AutoModelForCausalLM.from_pretrained("./aragpt2_lora_textgen")


    full_text = prompt

    for i in range(chunks):
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=chunk_size,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )
        new_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # أضف فقط النص الجديد بعد آخر طول للنص السابق
        full_text += new_text[len(full_text):]

    return full_text

# ✅ مثال استخدام
prompt = (
    "اكتب نص قانوني رسمي  لفصل موظف بشكل مفصل،في مصر "
)
long_text = generate_long_text(prompt, chunks=200, chunk_size=800)
print(long_text)
