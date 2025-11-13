"""
Inference pipeline for NewsSum project.
Use this script to generate summaries with trained models.
"""

from data_loader import load_and_prepare_data
from preprocessing import load_tokenizer
from model import initialize_model_for_inference, load_base_model, move_model_to_device
from inference import generate_summary, compare_model_outputs
from config import Config
import torch


def test_inference_on_sample():
    """
    Test inference on a sample article from the test set.
    """
    print("="*70)
    print("TESTING INFERENCE ON SAMPLE ARTICLE")
    print("="*70)
    
    # Load data
    print("\nLoading test data...")
    dataset = load_and_prepare_data()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(Config.FINETUNED_MODEL)
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    model = initialize_model_for_inference(use_finetuned=True)
    
    # Get sample article
    sample_idx = 50
    article = dataset['test'][sample_idx]['article']
    reference_summary = dataset['test'][sample_idx]['highlights']
    
    # Generate summary
    print("\nGenerating summary...")
    generated_summary = generate_summary(article, model, tokenizer)
    
    # Display results
    dash_line = '-' * 100
    print("\n" + dash_line)
    print('INPUT ARTICLE:')
    print(dash_line)
    print(article[:500] + "..." if len(article) > 500 else article)
    print(dash_line)
    print('REFERENCE SUMMARY:')
    print(dash_line)
    print(reference_summary)
    print(dash_line)
    print('GENERATED SUMMARY:')
    print(dash_line)
    print(generated_summary)
    print(dash_line)


def compare_models_on_sample():
    """
    Compare original and fine-tuned models on a sample article.
    """
    print("="*70)
    print("COMPARING ORIGINAL VS FINE-TUNED MODEL")
    print("="*70)
    
    # Load data
    print("\nLoading test data...")
    dataset = load_and_prepare_data()
    
    # Load tokenizers
    print("Loading tokenizers...")
    tokenizer = load_tokenizer()
    
    # Load models
    print("Loading original model...")
    original_model = load_base_model()
    device = Config.DEVICE if torch.cuda.is_available() else "cpu"
    original_model = move_model_to_device(original_model, device)
    original_model.eval()
    
    print("Loading fine-tuned model...")
    finetuned_model = initialize_model_for_inference(use_finetuned=True)
    
    # Get sample article
    sample_idx = 100
    article = dataset['test'][sample_idx]['article']
    reference_summary = dataset['test'][sample_idx]['highlights']
    
    # Compare outputs
    compare_model_outputs(
        article=article,
        original_model=original_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        reference_summary=reference_summary
    )


def generate_custom_summary(article_text):
    """
    Generate a summary for custom article text.
    
    Args:
        article_text: Custom article text to summarize.
    
    Returns:
        str: Generated summary.
    """
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer(Config.FINETUNED_MODEL)
    model = initialize_model_for_inference(use_finetuned=True)
    
    print("Generating summary...")
    summary = generate_summary(article_text, model, tokenizer)
    
    return summary


def main():
    """Main entry point for inference pipeline."""
    print("\nNewsSum Inference Pipeline")
    print("="*70)
    print("\nChoose an option:")
    print("1. Test on sample article")
    print("2. Compare original vs fine-tuned model")
    print("3. Enter custom article")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        test_inference_on_sample()
    elif choice == "2":
        compare_models_on_sample()
    elif choice == "3":
        print("\nEnter article text (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        article = "\n".join(lines)
        
        if article:
            summary = generate_custom_summary(article)
            print("\n" + "="*70)
            print("GENERATED SUMMARY:")
            print("="*70)
            print(summary)
            print("="*70)
        else:
            print("No article text provided!")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
