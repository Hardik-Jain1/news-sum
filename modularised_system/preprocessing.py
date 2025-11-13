"""
Preprocessing module for NewsSum project.
Handles tokenization and dataset preparation for training.
"""

from transformers import AutoTokenizer
from config import Config


def load_tokenizer(model_name=None):
    """
    Load the tokenizer for the specified model.
    
    Args:
        model_name: Name of the model to load tokenizer for. Defaults to Config.BASE_MODEL.
    
    Returns:
        tokenizer: Loaded tokenizer.
    """
    if model_name is None:
        model_name = Config.BASE_MODEL
    
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully!")
    return tokenizer


def tokenize_function(example, tokenizer):
    """
    Tokenize a single example from the dataset.
    
    Args:
        example: Dataset example containing 'article' and 'highlights'.
        tokenizer: Tokenizer to use for encoding.
    
    Returns:
        example: Tokenized example with 'input_ids' and 'labels'.
    """
    # Tokenize articles (input)
    example['input_ids'] = tokenizer(
        example["article"],
        max_length=Config.MAX_INPUT_LENGTH,
        truncation=True
    ).input_ids
    
    # Tokenize summaries (target/labels)
    example['labels'] = tokenizer(
        example["highlights"],
        max_length=Config.MAX_TARGET_LENGTH,
        truncation=True
    ).input_ids
    
    return example


def preprocess_dataset(dataset, tokenizer):
    """
    Preprocess the entire dataset by tokenizing all examples.
    
    Args:
        dataset: Raw dataset to preprocess.
        tokenizer: Tokenizer to use.
    
    Returns:
        tokenized_datasets: Preprocessed dataset ready for training.
    """
    print("\nPreprocessing dataset...")
    
    # Tokenize all examples in batches
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    # Remove original text columns (keep only tokenized data)
    tokenized_datasets = tokenized_datasets.remove_columns(['article', 'highlights', 'id'])
    
    print("Dataset preprocessing complete!")
    print(f"  - Training shape: {tokenized_datasets['train'].shape}")
    print(f"  - Validation shape: {tokenized_datasets['validation'].shape}")
    print(f"  - Test shape: {tokenized_datasets['test'].shape}")
    
    return tokenized_datasets


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_and_prepare_data
    
    data = load_and_prepare_data()
    tokenizer = load_tokenizer()
    tokenized_data = preprocess_dataset(data, tokenizer)
    
    print("\nSample tokenized data:")
    print(f"Input IDs length: {len(tokenized_data['train'][0]['input_ids'])}")
    print(f"Labels length: {len(tokenized_data['train'][0]['labels'])}")
