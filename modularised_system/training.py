"""
Training module for NewsSum project.
Handles model fine-tuning using Hugging Face Trainer.
"""

from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import torch
from config import Config


def setup_training_arguments():
    """
    Configure training arguments for the Seq2SeqTrainer.
    
    Returns:
        args: Configured training arguments.
    """
    print("Setting up training arguments...")
    
    args = Seq2SeqTrainingArguments(
        output_dir=f"./{Config.MODEL_NAME}-finetuned-cnn-news",
        evaluation_strategy=Config.EVALUATION_STRATEGY,
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        weight_decay=Config.WEIGHT_DECAY,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        num_train_epochs=Config.NUM_TRAIN_EPOCHS,
        predict_with_generate=Config.PREDICT_WITH_GENERATE,
        push_to_hub=Config.PUSH_TO_HUB,
        hub_strategy=Config.HUB_STRATEGY,
        logging_steps=100,
    )
    
    print("Training arguments configured successfully!")
    return args


def setup_data_collator(tokenizer, model):
    """
    Setup data collator for dynamic padding.
    
    Args:
        tokenizer: Tokenizer to use for padding.
        model: Model to use for collation.
    
    Returns:
        data_collator: Configured data collator.
    """
    print("Setting up data collator...")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    print("Data collator configured!")
    return data_collator


def setup_trainer(model, tokenizer, train_dataset, eval_dataset, compute_metrics_fn):
    """
    Setup the Seq2SeqTrainer with all necessary components.
    
    Args:
        model: Model to train.
        tokenizer: Tokenizer for the model.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        compute_metrics_fn: Function to compute evaluation metrics.
    
    Returns:
        trainer: Configured Seq2SeqTrainer.
    """
    print("\nSetting up trainer...")
    
    # Setup training arguments
    args = setup_training_arguments()
    
    # Setup data collator
    data_collator = setup_data_collator(tokenizer, model)
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
    )
    
    print("Trainer configured successfully!")
    return trainer


def clear_gpu_cache():
    """Clear GPU cache to prevent OOM errors."""
    if torch.cuda.is_available():
        print("Clearing GPU cache...")
        torch.cuda.empty_cache()
        print("GPU cache cleared!")


def train_model(trainer):
    """
    Train the model using the configured trainer.
    
    Args:
        trainer: Configured Seq2SeqTrainer.
    
    Returns:
        trainer: Trainer with trained model.
    """
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")
    
    # Clear GPU cache before training
    clear_gpu_cache()
    
    # Train the model
    trainer.train()
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70 + "\n")
    
    return trainer


def save_model(trainer, output_path=None):
    """
    Save the trained model locally.
    
    Args:
        trainer: Trainer with trained model.
        output_path: Path to save the model. Defaults to Config.OUTPUT_DIR.
    """
    if output_path is None:
        output_path = Config.OUTPUT_DIR
    
    print(f"Saving model to {output_path}...")
    trainer.save_model(output_path)
    print("Model saved successfully!")


def push_to_hub(trainer, commit_message="Training complete"):
    """
    Push the trained model to Hugging Face Hub.
    
    Args:
        trainer: Trainer with trained model.
        commit_message: Commit message for the push.
    """
    print(f"Pushing model to Hugging Face Hub...")
    trainer.push_to_hub(commit_message=commit_message, tags="summarization")
    print("Model pushed to Hub successfully!")


if __name__ == "__main__":
    print("Training module loaded successfully!")
    print("Use this module with the main pipeline to train the model.")
