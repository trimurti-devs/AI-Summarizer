import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from evaluate import load as load_metric
import os
from tqdm import tqdm

def load_model_and_tokenizer(model_dir):
    """
    Load the T5 model and tokenizer from the specified directory.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

def load_validation_data(csv_path):
    """
    Load the validation dataset from a CSV file and drop rows with missing values.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Validation CSV file not found: {csv_path}")
    val_df = pd.read_csv(csv_path).dropna()
    return val_df

def generate_summaries(model, tokenizer, texts, max_input_length=512, max_output_length=128, num_beams=4, batch_size=8):
    """
    Generate summaries for a list of texts using the T5 model and tokenizer with batch processing.
    """
    prefix = "summarize: "
    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating summaries"):
        batch_texts = [prefix + text for text in texts[i:i+batch_size]]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_input_length)
        output_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_output_length, num_beams=num_beams, early_stopping=True)
        batch_preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        predictions.extend(batch_preds)
    return predictions

def evaluate_summaries(predictions, references):
    """
    Evaluate generated summaries against reference summaries using ROUGE metric.
    """
    rouge = load_metric("rouge")
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    return results

def print_rouge_results(results):
    """
    Print ROUGE evaluation results.
    """
    print("\nROUGE Evaluation Results:")
    for key, value in results.items():
        # value is a float score
        print(f"{key.upper()}: {value:.4f}")

def main():
    model_dir = "C:/workspace-python/journal_POINTS_FIND_AI/output/t5_highlight_model"
    csv_path = "C:/workspace-python/journal_POINTS_FIND_AI/data/MixSub-SciHigh_val_FIRE.csv"

    try:
        model, tokenizer = load_model_and_tokenizer(model_dir)
        val_df = load_validation_data(csv_path)
    except FileNotFoundError as e:
        print(e)
        return

    abstracts = val_df["Abstract"].tolist()
    references = val_df["Highlights"].tolist()

    predictions = generate_summaries(model, tokenizer, abstracts)
    results = evaluate_summaries(predictions, references)
    print_rouge_results(results)

if __name__ == "__main__":
    main()
