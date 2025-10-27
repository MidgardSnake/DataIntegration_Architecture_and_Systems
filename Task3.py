import pandas as pd
from ollama import chat
from ollama import ChatResponse
import time
import re

#########1. Input Data #########
imdb_df = pd.read_csv('/Users/ludwigvoss/PyCharmMiscProject/00_input_data/imdb.csv', encoding='utf-8')
rt_df = pd.read_csv('/Users/ludwigvoss/PyCharmMiscProject/00_input_data/rotten_tomatoes.csv', encoding='utf-8')

#########2. Ground Truth #########
ground_truth = [
    ("Imdb.Name", "Rt.Name"),
    ("Imdb.YearRange", "Rt.Year"),
    ("Imdb.ReleaseDate", "Rt.Release Date"),
    ("Imdb.Director", "Rt.Director"),
    ("Imdb.Creator", "Rt.Creator"),
    ("Imdb.Cast", "Rt.Cast"),
    ("Imdb.Duration", "Rt.Duration"),
    ("Imdb.RatingValue", "Rt.RatingValue"),
    ("Imdb.Genre", "Rt.Genre"),
    ("Imdb.Description", "Rt.Description")
]

#########3. Sampling #########
imdb_sample = imdb_df.head(100).to_dict(orient='list')
rt_sample = rt_df.head(100).to_dict(orient='list')

keys_imdb = imdb_sample.keys()
keys_rt = rt_sample.keys()

#########4. Prompt #########
prompt = f"""
You are a data matching assistant. I will provide you with two datasets, each represented by a small sample of their columns.
Your task is to identify which columns from dataset A correspond to columns from dataset B based solely on the data values.

Dataset A: {imdb_sample}
Dataset B: {rt_sample}

Please return a Python dictionary called mapping, where keys are Dataset A column names and values are Dataset B column names.

The Example doesnt not correspond to real column names. 
DON'T INVENT NAMES, use the keys we provide in {keys_imdb} and {keys_rt}
Example:  

mapping = [
    ("Imdb.Age", "Rt.Birthyear"),
    ("Imdb.FirstName", "Rt.NameFirst"),
    ("Imdb.LastName", "Rt.FamilyName"),
    ...
]


print(mapping)
"""

##########4. Run LLM #########
start_time = time.time()
response: ChatResponse = chat(model='gemma3', messages=[{'role': 'user', 'content': prompt}])
end_time = time.time()

print("\n=== Raw LLM Output ===")
print(response.message.content)

##########5. Extract Mapping #########
text_output = response.message.content.strip()

# Entferne Codeblock-Markierungen (```python ... ```)
text_output = re.sub(r"^```python|```$", "", text_output, flags=re.MULTILINE).strip()

###########6. initialize Mapping

mapping = []

try:
    local_vars = {}
    exec(text_output, {}, local_vars)
    if 'mapping' in local_vars:
        temp_mapping = local_vars['mapping']
        # Wenn Dictionary, in Liste von Tuples umwandeln
        if isinstance(temp_mapping, dict):
            temp_mapping = list(temp_mapping.items())
        # Präfixe hinzufügen, nur für Spalten in ground_truth
        mapping = [
            (f"Imdb.{k}" if not k.startswith("Imdb.") else k,
             f"Rt.{v}" if not v.startswith("Rt.") else v)
            for k, v in temp_mapping
            if f"Imdb.{k}" in dict(ground_truth) or f"Imdb.{k}" not in dict(ground_truth)
        ]
except Exception as e:
    print("\n⚠️ Konnte mapping nicht ausführen:", e)
    print("Roher Output:\n", text_output)


########## Debug-output ##########
print("\n=== Parsed Mapping ===")
print(mapping)


##########7. Precision & Recall ##########
ground_truth_dict = dict(ground_truth)

if isinstance(mapping, list) and mapping:
    correct = sum(1 for k, v in mapping if ground_truth_dict.get(k) == v)
    precision = correct / len(mapping)
    recall = correct / len(ground_truth)
else:
    precision = recall = 0


print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Inference Time: {end_time - start_time:.2f} s")
print("Samples per column: 10")
