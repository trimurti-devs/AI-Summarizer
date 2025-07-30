from transformers import T5Tokenizer, T5ForConditionalGeneration

model_dir = "C:/workspace-python/journal_POINTS_FIND_AI/output/t5_highlight_model"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

import sys
import argparse

parser = argparse.ArgumentParser(description="Generate highlight from abstract")
parser.add_argument('--abstract', type=str, nargs='+', help='Abstract text input')
parser.add_argument('--file', type=str, help='Path to file containing abstract text')

args = parser.parse_args()

if args.abstract:
    abstract = " ".join(args.abstract)
elif args.file:
    with open(args.file, 'r', encoding='utf-8') as f:
        abstract = f.read()
else:
    print("Error: Please provide abstract text via --abstract or --file argument")
    sys.exit(1)

print("Abstract input:", abstract)

input_ids = tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512).input_ids
output_ids = model.generate(
    input_ids,
    max_length=128,
    num_beams=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    early_stopping=True
)
highlight = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Highlight:", highlight)
