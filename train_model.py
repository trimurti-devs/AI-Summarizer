import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# Config
MODEL_NAME = "t5-base"
TRAIN_PATH = "C:/workspace-python/journal_POINTS_FIND_AI/data/MixSub-SciHigh_train_FIRE.csv"
VAL_PATH = "C:/workspace-python/journal_POINTS_FIND_AI/data/MixSub-SciHigh_val_FIRE.csv"
OUTPUT_DIR = "C:/workspace-python/journal_POINTS_FIND_AI/output/t5_highlight_model"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[["Abstract", "Highlights"]].dropna()
    df = df.rename(columns={"Abstract": "input", "Highlights": "output"})
    return Dataset.from_pandas(df)

train_dataset = load_data(TRAIN_PATH)
val_dataset = load_data(VAL_PATH)

# Tokenizer and Model
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Tokenize
def preprocess(example):
    prefix = "summarize: "
    input_enc = tokenizer(
        prefix + example["input"], padding="max_length", truncation=True,
        max_length=MAX_INPUT_LENGTH
    )
    target_enc = tokenizer(
        example["output"], padding="max_length", truncation=True,
        max_length=MAX_TARGET_LENGTH
    )
    input_enc["labels"] = [
        (label if label != tokenizer.pad_token_id else -100)
        for label in target_enc["input_ids"]
    ]
    return input_enc

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=torch.cuda.is_available(),
    logging_dir='./logs',
    logging_steps=25,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model)
)

# Train
trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
