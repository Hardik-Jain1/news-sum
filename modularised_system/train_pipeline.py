"""
Main pipeline for NewsSum project.
Orchestrates the complete training and evaluation workflow.
"""

from data_loader import load_and_prepare_data
from preprocessing import load_tokenizer, preprocess_dataset
from model import initialize_model_for_training, print_model_parameters
from evaluation import create_compute_metrics_function, evaluate_model
from training import setup_trainer, train_model, push_to_hub
from config import Config


def run_training_pipeline():
    """
    Execute the complete training pipeline.
    
    This includes:
    1. Loading and preparing data
    2. Loading tokenizer
    3. Preprocessing dataset
    4. Initializing model
    5. Setting up trainer
    6. Training model
    7. Evaluating model
    8. Pushing to Hugging Face Hub
    """
    print("="*70)
    print("STARTING NEWSSUM TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Load data
    print("\n[STEP 1/7] Loading and preparing data...")
    dataset = load_and_prepare_data()
    
    # Step 2: Load tokenizer
    print("\n[STEP 2/7] Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    # Step 3: Preprocess dataset
    print("\n[STEP 3/7] Preprocessing dataset...")
    tokenized_datasets = preprocess_dataset(dataset, tokenizer)
    
    # Step 4: Initialize model
    print("\n[STEP 4/7] Initializing model...")
    model = initialize_model_for_training()
    
    # Step 5: Create compute metrics function
    print("\n[STEP 5/7] Setting up evaluation metrics...")
    compute_metrics = create_compute_metrics_function(tokenizer)
    
    # Step 6: Setup and run training
    print("\n[STEP 6/7] Setting up trainer and training model...")
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics_fn=compute_metrics
    )
    
    trainer = train_model(trainer)
    
    # Step 7: Evaluate model
    print("\n[STEP 7/7] Evaluating model...")
    results = evaluate_model(trainer)
    
    # Optional: Push to Hub
    if Config.PUSH_TO_HUB:
        print("\nPushing model to Hugging Face Hub...")
        push_to_hub(trainer, commit_message="Training complete")
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*70)
    
    return trainer, results


def main():
    """Main entry point for the training pipeline."""
    try:
        trainer, results = run_training_pipeline()
        print("\n✅ Pipeline executed successfully!")
        return trainer, results
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
