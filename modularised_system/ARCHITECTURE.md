# NewsSum - Modular Architecture

## Project Structure

```
news-sum/modularised_system/
│
├── config.py                 # Configuration and hyperparameters
├── data_loader.py           # Data loading and sampling
├── preprocessing.py         # Tokenization and preprocessing
├── model.py                 # Model initialization and setup
├── training.py              # Training logic and utilities
├── evaluation.py            # Evaluation metrics and comparison
├── inference.py             # Inference and summary generation
├── train_pipeline.py        # Main training pipeline
├── inference_pipeline.py    # Main inference pipeline
├── requirements.txt         # Python dependencies
└── ARCHITECTURE.md          # Project artchitecture and module overview
```

## Module Overview

### 1. **config.py**
Central configuration file containing all hyperparameters and settings:
- Model configuration (base model, paths)
- Tokenization parameters (max lengths)
- Dataset configuration (sampling ratios)
- Training hyperparameters (batch size, learning rate, epochs)
- Device and evaluation settings

### 2. **data_loader.py**
Handles data loading and preparation:
- `load_cnn_dailymail()`: Load the full dataset from Hugging Face
- `sample_dataset()`: Create sampled subsets for efficient training
- `load_and_prepare_data()`: Complete data loading pipeline

### 3. **preprocessing.py**
Data preprocessing and tokenization:
- `load_tokenizer()`: Load the appropriate tokenizer
- `tokenize_function()`: Tokenize articles and summaries
- `preprocess_dataset()`: Preprocess entire dataset in batches

### 4. **model.py**
Model initialization and management:
- `load_base_model()`: Load pre-trained BART model
- `load_finetuned_model()`: Load fine-tuned model from Hub
- `move_model_to_device()`: Handle GPU/CPU placement
- `print_model_parameters()`: Display parameter information
- `initialize_model_for_training()`: Setup for training
- `initialize_model_for_inference()`: Setup for inference

### 5. **training.py**
Training configuration and execution:
- `setup_training_arguments()`: Configure training parameters
- `setup_data_collator()`: Setup dynamic padding
- `setup_trainer()`: Initialize Seq2SeqTrainer
- `train_model()`: Execute training loop
- `save_model()`: Save trained model
- `push_to_hub()`: Upload to Hugging Face Hub

### 6. **evaluation.py**
Model evaluation and metrics:
- `load_rouge_metric()`: Load ROUGE metric
- `create_compute_metrics_function()`: Create metrics function for training
- `evaluate_model()`: Evaluate trained model
- `compare_models()`: Compare original vs fine-tuned performance

### 7. **inference.py**
Summary generation:
- `generate_summary()`: Generate summary for single article
- `generate_summary_with_beam_search()`: Use beam search for better quality
- `display_comparison()`: Format and display results
- `batch_generate_summaries()`: Generate for multiple articles
- `compare_model_outputs()`: Compare original vs fine-tuned outputs

### 8. **train_pipeline.py**
Main training pipeline orchestrating all modules:
- Complete end-to-end training workflow
- Step-by-step execution with progress reporting
- Error handling and completion status

### 9. **inference_pipeline.py**
Main inference pipeline for generating summaries:
- Test on sample articles
- Compare original vs fine-tuned models
- Generate summaries for custom text
- Interactive command-line interface

## Usage

### Training

```bash
# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python train_pipeline.py
```

### Inference

```bash
# Run inference pipeline (interactive)
python inference_pipeline.py

# Or use programmatically
from inference_pipeline import generate_custom_summary

article = "Your article text here..."
summary = generate_custom_summary(article)
print(summary)
```

### Individual Modules

Each module can be used independently:

```python
# Example: Load data only
from data_loader import load_and_prepare_data
dataset = load_and_prepare_data()

# Example: Generate summary
from inference import generate_summary
from model import initialize_model_for_inference
from preprocessing import load_tokenizer

tokenizer = load_tokenizer("hardikJ11/bart-base-finetuned-cnn-news")
model = initialize_model_for_inference(use_finetuned=True)
summary = generate_summary(article, model, tokenizer)
```

## Benefits of Modular Design

1. **Maintainability**: Each module has a single responsibility
2. **Reusability**: Functions can be imported and used across projects
3. **Testability**: Individual components can be tested in isolation
4. **Scalability**: Easy to extend or modify specific parts
5. **Readability**: Clear separation of concerns
6. **Flexibility**: Mix and match components as needed

## Configuration

All settings can be modified in `config.py` without touching the core logic:

```python
class Config:
    BASE_MODEL = "facebook/bart-base"
    MAX_INPUT_LENGTH = 1024
    BATCH_SIZE = 8
    NUM_TRAIN_EPOCHS = 5
    # ... and more
```

## Next Steps

1. Adjust hyperparameters in `config.py`
2. Run training with `python train_pipeline.py`
3. Test inference with `python inference_pipeline.py`
4. Integrate modules into your own workflows
5. Extend functionality by adding new modules

## Notes

- The import errors for `torch`, `transformers`, etc. are expected if dependencies aren't installed yet
- Run `pip install -r requirements.txt` to resolve these
- The original notebook (`NewsSum.ipynb`) is kept as reference
- All modules include `__main__` blocks for testing individual functionality
