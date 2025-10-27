import pandas as pd
from ollama import chat
from ollama import ChatResponse
import time

# === Data ===
imdb_df = pd.read_csv('/Users/ludwigvoss/PyCharmMiscProject/00_input_data/imdb.csv',encoding='utf-8')
rt_df = pd.read_csv('/Users/ludwigvoss/PyCharmMiscProject/00_input_data/rotten_tomatoes.csv',encoding='utf-8')


# === Ground Truth ===
G = [
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
# Nimm z.B. die ersten 10 Werte jeder Spalte als Sample
imdb_sample = imdb_df.head(10).to_dict(orient='list')
rt_sample = rt_df.head(10).to_dict(orient='list')

prompt = f"""
You are a data matching assistant. I will provide you with two datasets, each represented by a small sample of their columns.
Your task is to identify which columns from dataset A correspond to columns from dataset B based solely on the data values.

Dataset A: {imdb_sample}
Dataset B: {rt_sample}

Please return a mapping in the format:
{{
    "Dataset A Column Name": "Dataset B Column Name",
    ...
}}
"""

start_time = time.time()
response: ChatResponse = chat(model='gemma3', messages=[
  {
    'role': 'user',
    'content': prompt,
  },
])
end_time = time.time()
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)

print(f"Duration: {str(end_time - start_time)}")
