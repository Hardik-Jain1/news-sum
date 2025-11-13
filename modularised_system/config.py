"""
Configuration file for NewsSum project.
Contains all hyperparameters and model settings.
"""

class Config:
    """Central configuration for the NewsSum project."""
    
    # Model Configuration
    BASE_MODEL = "facebook/bart-base"
    FINETUNED_MODEL = "hardikJ11/bart-base-finetuned-cnn-news"
    
    # Tokenization Configuration
    MAX_INPUT_LENGTH = 1024
    MAX_TARGET_LENGTH = 150
    
    # Dataset Configuration
    DATASET_NAME = "cnn_dailymail"
    DATASET_VERSION = "3.0.0"
    TRAIN_SHARD = 50  # Use 1/50th of training data
    VALIDATION_SHARD = 40  # Use 1/40th of validation data
    TEST_SHARD = 40  # Use 1/40th of test data
    
    # Training Configuration
    BATCH_SIZE = 8
    NUM_TRAIN_EPOCHS = 5
    LEARNING_RATE = 5.6e-4
    WEIGHT_DECAY = 0.01
    SAVE_TOTAL_LIMIT = 3
    
    # Output Configuration
    OUTPUT_DIR = "./bart-base-finetuned-cnn-news"
    MODEL_NAME = "bart-base"
    
    # Hugging Face Hub Configuration
    PUSH_TO_HUB = True
    HUB_STRATEGY = "every_save"
    
    # Device Configuration
    DEVICE = "cuda:0"
    
    # Evaluation Configuration
    PREDICT_WITH_GENERATE = True
    EVALUATION_STRATEGY = "epoch"
