"""
Model initialization module for NewsSum project.
Handles loading and preparing models for training and inference.
"""

from transformers import AutoModelForSeq2SeqLM
import torch
from config import Config


def load_base_model(model_name=None):
    """
    Load the base pre-trained model.
    
    Args:
        model_name: Name of the model to load. Defaults to Config.BASE_MODEL.
    
    Returns:
        model: Loaded model ready for fine-tuning.
    """
    if model_name is None:
        model_name = Config.BASE_MODEL
    
    print(f"Loading base model: {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Model loaded successfully!")
    
    return model


def load_finetuned_model(model_name=None):
    """
    Load a fine-tuned model from Hugging Face Hub.
    
    Args:
        model_name: Name of the fine-tuned model. Defaults to Config.FINETUNED_MODEL.
    
    Returns:
        model: Loaded fine-tuned model.
    """
    if model_name is None:
        model_name = Config.FINETUNED_MODEL
    
    print(f"Loading fine-tuned model: {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Fine-tuned model loaded successfully!")
    
    return model


def move_model_to_device(model, device=None):
    """
    Move model to specified device (GPU/CPU).
    
    Args:
        model: Model to move.
        device: Device to move to. Defaults to Config.DEVICE.
    
    Returns:
        model: Model on the specified device.
    """
    if device is None:
        device = Config.DEVICE
    
    if torch.cuda.is_available() and "cuda" in device:
        print(f"Moving model to {device}...")
        model = model.to(device)
        print("Model moved to GPU successfully!")
    else:
        print("CUDA not available. Using CPU...")
        model = model.to("cpu")
    
    return model


def print_model_parameters(model):
    """
    Print detailed information about model parameters.
    
    Args:
        model: Model to analyze.
    
    Returns:
        str: Formatted string with parameter information.
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percentage = 100 * trainable_params / all_params if all_params > 0 else 0
    
    info = (
        f"\n{'='*50}\n"
        f"Model Parameters:\n"
        f"{'='*50}\n"
        f"Trainable parameters: {trainable_params:,}\n"
        f"All parameters: {all_params:,}\n"
        f"Trainable percentage: {percentage:.2f}%\n"
        f"{'='*50}"
    )
    
    print(info)
    return info


def initialize_model_for_training():
    """
    Initialize and prepare model for training.
    
    Returns:
        model: Model ready for training.
    """
    model = load_base_model()
    print_model_parameters(model)
    return model


def initialize_model_for_inference(use_finetuned=True, device=None):
    """
    Initialize and prepare model for inference.
    
    Args:
        use_finetuned: Whether to use fine-tuned model or base model.
        device: Device to load model on.
    
    Returns:
        model: Model ready for inference.
    """
    if use_finetuned:
        model = load_finetuned_model()
    else:
        model = load_base_model()
    
    model = move_model_to_device(model, device)
    model.eval()  # Set to evaluation mode
    
    return model


if __name__ == "__main__":
    # Test model initialization
    print("Testing base model initialization...")
    model = initialize_model_for_training()
    
    print("\nTesting fine-tuned model initialization...")
    finetuned_model = initialize_model_for_inference(use_finetuned=True)
