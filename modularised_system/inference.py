"""
Inference module for NewsSum project.
Handles generating summaries using trained models.
"""

import torch
from transformers import GenerationConfig
from config import Config


def generate_summary(article, model, tokenizer, max_new_tokens=None, device=None):
    """
    Generate a summary for a given article.
    
    Args:
        article: Input article text.
        model: Model to use for generation.
        tokenizer: Tokenizer for encoding/decoding.
        max_new_tokens: Maximum number of tokens to generate.
        device: Device to run inference on.
    
    Returns:
        str: Generated summary.
    """
    if max_new_tokens is None:
        max_new_tokens = Config.MAX_TARGET_LENGTH
    
    if device is None:
        device = Config.DEVICE if torch.cuda.is_available() else "cpu"
    
    # Tokenize input
    inputs = tokenizer(
        article,
        max_length=Config.MAX_INPUT_LENGTH,
        return_tensors="pt",
        truncation=True
    )
    
    # Move to device
    input_ids = inputs["input_ids"].to(device)
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
        )
    
    # Decode output
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary


def generate_summary_with_beam_search(article, model, tokenizer, num_beams=4, device=None):
    """
    Generate a summary using beam search for better quality.
    
    Args:
        article: Input article text.
        model: Model to use for generation.
        tokenizer: Tokenizer for encoding/decoding.
        num_beams: Number of beams for beam search.
        device: Device to run inference on.
    
    Returns:
        str: Generated summary.
    """
    if device is None:
        device = Config.DEVICE if torch.cuda.is_available() else "cpu"
    
    # Tokenize input
    inputs = tokenizer(
        article,
        max_length=Config.MAX_INPUT_LENGTH,
        return_tensors="pt",
        truncation=True
    )
    
    input_ids = inputs["input_ids"].to(device)
    
    # Configure generation
    generation_config = GenerationConfig(
        max_new_tokens=Config.MAX_TARGET_LENGTH,
        num_beams=num_beams
    )
    
    # Generate with beam search
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config
        )
    
    # Decode output
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary


def display_comparison(article, reference_summary, generated_summary, model_name="Model"):
    """
    Display a formatted comparison of article, reference, and generated summary.
    
    Args:
        article: Original article text.
        reference_summary: Human-written reference summary.
        generated_summary: Model-generated summary.
        model_name: Name of the model for display.
    """
    dash_line = '-' * 100
    
    print(dash_line)
    print(f'INPUT ARTICLE:')
    print(dash_line)
    print(article[:500] + "..." if len(article) > 500 else article)
    print(dash_line)
    print(f'REFERENCE SUMMARY:')
    print(dash_line)
    print(reference_summary)
    print(dash_line)
    print(f'{model_name.upper()} SUMMARY:')
    print(dash_line)
    print(generated_summary)
    print(dash_line)
    print()


def batch_generate_summaries(articles, model, tokenizer, device=None):
    """
    Generate summaries for a batch of articles.
    
    Args:
        articles: List of article texts.
        model: Model to use for generation.
        tokenizer: Tokenizer for encoding/decoding.
        device: Device to run inference on.
    
    Returns:
        list: List of generated summaries.
    """
    if device is None:
        device = Config.DEVICE if torch.cuda.is_available() else "cpu"
    
    summaries = []
    
    print(f"Generating summaries for {len(articles)} articles...")
    
    for idx, article in enumerate(articles):
        summary = generate_summary(article, model, tokenizer, device=device)
        summaries.append(summary)
        
        if (idx + 1) % 10 == 0:
            print(f"  Generated {idx + 1}/{len(articles)} summaries")
    
    print(f"All summaries generated!")
    
    return summaries


def compare_model_outputs(article, original_model, finetuned_model, tokenizer, reference_summary=None):
    """
    Compare outputs from original and fine-tuned models on the same article.
    
    Args:
        article: Input article text.
        original_model: Original base model.
        finetuned_model: Fine-tuned model.
        tokenizer: Tokenizer for both models.
        reference_summary: Optional human reference summary.
    """
    device = Config.DEVICE if torch.cuda.is_available() else "cpu"
    
    print("\nGenerating summaries...")
    
    # Generate with original model
    original_summary = generate_summary(article, original_model, tokenizer, device=device)
    
    # Generate with fine-tuned model
    finetuned_summary = generate_summary(article, finetuned_model, tokenizer, device=device)
    
    # Display results
    dash_line = '-' * 100
    print(dash_line)
    print('INPUT ARTICLE:')
    print(dash_line)
    print(article[:500] + "..." if len(article) > 500 else article)
    
    if reference_summary:
        print(dash_line)
        print('REFERENCE SUMMARY:')
        print(dash_line)
        print(reference_summary)
    
    print(dash_line)
    print('ORIGINAL MODEL SUMMARY:')
    print(dash_line)
    print(original_summary)
    print(dash_line)
    print('FINE-TUNED MODEL SUMMARY:')
    print(dash_line)
    print(finetuned_summary)
    print(dash_line)
    print()


if __name__ == "__main__":
    print("Inference module loaded successfully!")
    print("Use this module with the main pipeline to generate summaries.")
