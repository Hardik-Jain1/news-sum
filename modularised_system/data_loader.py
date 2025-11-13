"""
Data loading module for NewsSum project.
Handles loading and sampling of the CNN/DailyMail dataset.
"""

from datasets import load_dataset
from config import Config


def load_cnn_dailymail():
    """
    Load the CNN/DailyMail dataset from Hugging Face.
    
    Returns:
        dataset: Full CNN/DailyMail dataset with train, validation, and test splits.
    """
    print(f"Loading {Config.DATASET_NAME} dataset (version {Config.DATASET_VERSION})...")
    dataset = load_dataset(Config.DATASET_NAME, Config.DATASET_VERSION)
    print(f"Dataset loaded successfully!")
    print(f"  - Train samples: {len(dataset['train'])}")
    print(f"  - Validation samples: {len(dataset['validation'])}")
    print(f"  - Test samples: {len(dataset['test'])}")
    return dataset


def sample_dataset(dataset):
    """
    Sample a subset of the dataset for efficient training.
    
    Args:
        dataset: Full dataset to sample from.
    
    Returns:
        sampled_dataset: Sampled dataset with reduced splits.
    """
    print("\nSampling dataset for efficient training...")
    
    # Shuffle and sample
    dataset = dataset.shuffle()
    
    sampled_dataset = {
        'train': dataset['train'].shard(num_shards=Config.TRAIN_SHARD, index=0),
        'validation': dataset['validation'].shard(num_shards=Config.VALIDATION_SHARD, index=0),
        'test': dataset['test'].shard(num_shards=Config.TEST_SHARD, index=0)
    }
    
    print(f"Dataset sampled successfully!")
    print(f"  - Train samples: {len(sampled_dataset['train'])}")
    print(f"  - Validation samples: {len(sampled_dataset['validation'])}")
    print(f"  - Test samples: {len(sampled_dataset['test'])}")
    
    return sampled_dataset


def load_and_prepare_data():
    """
    Load and prepare the dataset for training.
    
    Returns:
        sampled_dataset: Prepared dataset ready for preprocessing.
    """
    dataset = load_cnn_dailymail()
    sampled_dataset = sample_dataset(dataset)
    return sampled_dataset


if __name__ == "__main__":
    # Test the data loading
    data = load_and_prepare_data()
    print("\nSample article:")
    print(data['train'][0]['article'][:200] + "...")
    print("\nSample summary:")
    print(data['train'][0]['highlights'])
