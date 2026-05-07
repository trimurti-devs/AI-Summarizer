# AI-Summarizer

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

## Overview

AI-Summarizer is an intelligent text summarization system powered by state-of-the-art transformer models. It automatically extracts and generates concise summaries from lengthy academic abstracts and documents, identifying and preserving the most critical information while reducing verbosity.

Built on the T5 (Text-to-Text Transfer Transformer) architecture, this system is specifically optimized for scientific text summarization and highlight extraction.

## Features

- **Automatic Text Summarization**: Advanced neural network-based summarization using T5 transformer model
- **Batch Processing**: Efficient processing of large datasets with configurable batch sizes
- **Customizable Parameters**: Fine-tune model behavior including input/output length, beam search parameters, and sampling strategies
- **Comprehensive Evaluation**: Built-in metrics including ROUGE scores for rigorous performance assessment
- **Command-Line Interface**: Flexible input handling via direct text or file input
- **GPU Acceleration**: Automatic CUDA support with mixed-precision training for faster processing

## Architecture

This project implements a seq2seq (sequence-to-sequence) architecture using:
- **Model**: T5-base (pre-trained on 750GB of text data)
- **Framework**: PyTorch with Hugging Face Transformers
- **Evaluation Metrics**: ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **Input**: Academic abstracts (up to 512 tokens)
- **Output**: Key highlights/summaries (up to 128 tokens)

## Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version
- **CUDA** (optional): For GPU acceleration
- **System Memory**: Minimum 8GB RAM (16GB recommended for efficient training)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/trimurti-devs/AI-Summarizer.git
cd AI-Summarizer
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The system expects CSV files with the following structure:

| Abstract | Highlights |
|----------|-----------|
| Full academic text... | Key points summary... |
| ... | ... |

Update the dataset paths in the respective scripts:
- Training data path: Line 8 in `train_model.py`
- Validation data path: Line 9 in `train_model.py`
- Output directory: Line 10 in `train_model.py`

## Usage

### 1. Prepare and Verify Data

Inspect your dataset and verify tokenization:

```bash
python prepare_data.py
```

This script displays sample entries and tokenized representations to ensure data integrity.

### 2. Train the Model

Start the training process with your prepared dataset:

```bash
python train_model.py
```

**Training Configuration:**
- Epochs: 5
- Batch Size: 4 (train/eval)
- Learning Rate: 5e-5 (default)
- Optimization Strategy: Evaluation and checkpoint saving at each epoch
- GPU Support: Automatically enabled if CUDA is available

**Output**: Trained model and tokenizer saved to the configured `OUTPUT_DIR`

### 3. Evaluate Model Performance

Assess the trained model using standard evaluation metrics:

```bash
python evaluate_model.py
```

**Metrics Generated:**
- **ROUGE-1**: Unigram overlap between predicted and reference summaries
- **ROUGE-2**: Bigram overlap for capturing phrase-level accuracy
- **ROUGE-L**: Longest common subsequence for capturing semantic structure

### 4. Test with Sample Input

Generate summaries from text input:

**Option A: Direct Text Input**
```bash
python test.py --abstract "This is a sample abstract text about machine learning..."
```

**Option B: Input from File**
```bash
python test.py --file path/to/abstract.txt
```

**Generation Parameters:**
- Beam Search: 2 beams
- Sampling: Top-k=50, Top-p=0.95
- Max Output Length: 128 tokens
- Early Stopping: Enabled

## Project Structure

```
AI-Summarizer/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── train_model.py              # Model training pipeline
├── evaluate_model.py           # Model evaluation with metrics
├── prepare_data.py             # Data preparation and verification
├── test.py                     # Inference script for text summarization
├── data/                       # Dataset directory (CSV files)
└── output/                     # Trained model artifacts
```

## File Descriptions

| File | Purpose |
|------|---------|
| **train_model.py** | Trains the T5 model on prepared dataset with validation |
| **evaluate_model.py** | Generates summaries and computes ROUGE evaluation metrics |
| **prepare_data.py** | Loads, inspects, and tokenizes raw data for training |
| **test.py** | Command-line interface for generating summaries from input text |
| **requirements.txt** | Lists all required Python packages and versions |

## Dependencies

Core libraries and their purposes:

```
PyTorch>=2.0.0              # Deep learning framework
transformers>=4.30.0        # Pre-trained models and utilities
datasets>=2.10.0            # Dataset loading and processing
pandas>=1.5.0               # Data manipulation
evaluate>=0.4.0             # Metric computation
torch-cuda                  # GPU acceleration (optional)
```

For detailed requirements, see `requirements.txt`.

## Model Configuration

### Training Parameters
- **Model**: T5-base (220M parameters)
- **Input Max Length**: 512 tokens
- **Output Max Length**: 128 tokens
- **Optimization**: AdamW with learning rate scheduling
- **Loss**: Cross-entropy with label smoothing

### Inference Parameters
- **Beam Search Width**: 2-4 beams recommended
- **Temperature**: 1.0 (default)
- **Sampling**: Top-k and Top-p sampling for diversity
- **Early Stopping**: Enabled to reduce computation

## Performance

Expected performance on scientific text datasets:

| Metric | Range |
|--------|-------|
| ROUGE-1 | 0.40-0.55 |
| ROUGE-2 | 0.18-0.35 |
| ROUGE-L | 0.38-0.52 |

*Note: Actual performance depends on dataset quality and training duration*

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` in `train_model.py`
- Enable mixed-precision training (fp16)
- Use gradient accumulation

### Model Not Found Error
- Verify paths in scripts match your system configuration
- Ensure model output directory exists and is writable

### Dataset Loading Issues
- Confirm CSV files have "Abstract" and "Highlights" columns
- Check for missing or malformed data entries
- Verify file encoding is UTF-8

## Future Enhancements

- [ ] Support for multiple languages
- [ ] Fine-tuning with domain-specific datasets
- [ ] Real-time API endpoint
- [ ] Web-based interface
- [ ] Model quantization for edge deployment
- [ ] Comparison with alternative models (BART, Pegasus)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{aisummarizer2025,
  title={AI-Summarizer: Automated Text Summarization System},
  author={Trimurti Devs},
  year={2025},
  url={https://github.com/trimurti-devs/AI-Summarizer}
}
```

## Acknowledgments

- Hugging Face for the Transformers library and pre-trained models
- Google Research for the T5 architecture
- The open-source community for PyTorch and related tools

## Support

For issues, questions, or feedback:
- Open an [Issue](https://github.com/trimurti-devs/AI-Summarizer/issues)
- Check existing documentation
- Review the [Discussions](https://github.com/trimurti-devs/AI-Summarizer/discussions)

---

**Last Updated**: May 2026  
**Status**: Active Development
