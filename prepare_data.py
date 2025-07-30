# prepare_data.py
import pandas as pd
from transformers import AutoTokenizer

# Choose your model (can also try "facebook/bart-large-cnn" or "google/pegasus-xsum")
MODEL_NAME = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load CSVs
train_df = pd.read_csv("C:/workspace-python/journal_POINTS_FIND_AI/data/MixSub-SciHigh_train_FIRE.csv")
val_df = pd.read_csv("C:/workspace-python/journal_POINTS_FIND_AI/data/MixSub-SciHigh_val_FIRE.csv")

# Show sample
print("Sample Train Entry:")
print("Abstract:", train_df['Abstract'].iloc[0])
print("Highlights:", train_df['Highlights'].iloc[0])

# Tokenize sample
sample_input = train_df['Abstract'].iloc[0]
sample_target = train_df['Highlights'].iloc[0]

inputs = tokenizer(sample_input, max_length=512, truncation=True, return_tensors="pt")
labels = tokenizer(sample_target, max_length=128, truncation=True, return_tensors="pt")

print("\nTokenized Input IDs:", inputs['input_ids'])
print("Tokenized Label IDs:", labels['input_ids'])
