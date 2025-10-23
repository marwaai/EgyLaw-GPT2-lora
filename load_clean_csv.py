import pandas as pd
import re
from datasets import load_dataset

# قراءة الملف
data = pd.read_csv(r".\all_legal_articles.csv")

# نمط regex لإزالة الأرقام الإنجليزية والهندية
pattern = r"[^ء-ي]"

# إنشاء قائمة جديدة بعد التنظيف
t = []

for text in data["Text"]:
    if   len(str(text))>2:
        # تأكدي أن القيمة نصية
        text = str(text)
        # إزالة الأرقام
        text = re.sub(pattern, " ", text)
        text = re.sub(r"\s+"," ", text).strip()

        t.append(text)

# عرض أول 5 أسطر بعد التنظيف
print(t)
df = pd.DataFrame({"text": t})
df.to_json("cleaned_texts.json", orient="records", force_ascii=False)

dataset = load_dataset("json", data_files="cleaned_texts.json")

