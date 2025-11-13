"""
Evaluation module for NewsSum project.
Handles model evaluation using ROUGE metrics.
"""

import evaluate
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from config import Config


# Download NLTK punkt tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')


def load_rouge_metric():
    """
    Load the ROUGE evaluation metric.
    
    Returns:
        rouge: ROUGE metric object.
    """
    print("Loading ROUGE metric...")
    rouge = evaluate.load('rouge')
    print("ROUGE metric loaded!")
    return rouge


def create_compute_metrics_function(tokenizer):
    """
    Create a compute_metrics function for use during training.
    
    Args:
        tokenizer: Tokenizer to decode predictions.
    
    Returns:
        compute_metrics: Function that computes ROUGE metrics.
    """
    rouge_score = load_rouge_metric()
    
    def compute_metrics(eval_pred):
        """
        Compute ROUGE metrics for evaluation.
        
        Args:
            eval_pred: Tuple of (predictions, labels).
        
        Returns:
            dict: Dictionary of ROUGE scores.
        """
        predictions, labels = eval_pred
        
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels (can't decode them)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode reference summaries
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Add newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        
        # Compute ROUGE scores
        result = rouge_score.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # Extract F1 scores as percentages
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        return {k: round(v, 4) for k, v in result.items()}
    
    return compute_metrics


def evaluate_model(trainer):
    """
    Evaluate the model using the trainer's evaluation dataset.
    
    Args:
        trainer: Trained Seq2SeqTrainer.
    
    Returns:
        dict: Evaluation results.
    """
    print("\n" + "="*70)
    print("Evaluating model...")
    print("="*70 + "\n")
    
    results = trainer.evaluate()
    
    print("\nEvaluation Results:")
    print("-" * 50)
    for key, value in results.items():
        print(f"{key}: {value}")
    print("-" * 50 + "\n")
    
    return results


def compare_models(original_summaries, finetuned_summaries, reference_summaries):
    """
    Compare original and fine-tuned model outputs using ROUGE.
    
    Args:
        original_summaries: Summaries from the original model.
        finetuned_summaries: Summaries from the fine-tuned model.
        reference_summaries: Human reference summaries.
    
    Returns:
        tuple: (original_results, finetuned_results, improvement)
    """
    rouge = load_rouge_metric()
    
    print("\n" + "="*70)
    print("Comparing models...")
    print("="*70 + "\n")
    
    # Evaluate original model
    original_results = rouge.compute(
        predictions=original_summaries,
        references=reference_summaries[:len(original_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )
    
    # Evaluate fine-tuned model
    finetuned_results = rouge.compute(
        predictions=finetuned_summaries,
        references=reference_summaries[:len(finetuned_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )
    
    # Calculate improvement
    improvement = {}
    for key in finetuned_results.keys():
        if key in original_results:
            orig_val = original_results[key]
            ft_val = finetuned_results[key]
            improvement[key] = ((ft_val - orig_val) / orig_val) * 100 if orig_val > 0 else 0
    
    # Print results
    print("ORIGINAL MODEL:")
    for key, value in original_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nFINE-TUNED MODEL:")
    for key, value in finetuned_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nIMPROVEMENT:")
    for key, value in improvement.items():
        print(f"  {key}: {value:+.2f}%")
    
    print("="*70 + "\n")
    
    return original_results, finetuned_results, improvement


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("Use this module with the main pipeline to evaluate models.")
