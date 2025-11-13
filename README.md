# NewsSum: Neural Text Summarization with Fine-Tuned BART

An instructional Large Language Model (LLM) designed for automated news article summarization. This project leverages BART (Bidirectional and Auto-Regressive Transformers) fine-tuned on the CNN/DailyMail dataset, achieving a **15.85% performance improvement** over the baseline model as measured by ROUGE metrics.

## 🎯 Project Overview

NewsSum demonstrates end-to-end ML engineering practices for natural language processing, including:
- Full fine-tuning of a seq2seq transformer model
- Custom data preprocessing and tokenization pipelines
- Quantitative evaluation using industry-standard ROUGE metrics
- Model deployment to Hugging Face Hub
- Comparative analysis between baseline and fine-tuned models

## 🏗️ Architecture

**Base Model:** `facebook/bart-base`
- **Model Type:** Sequence-to-Sequence Transformer (Encoder-Decoder)
- **Parameters:** ~139M trainable parameters
- **Input Length:** 1024 tokens (max)
- **Output Length:** 150 tokens (max)

**Dataset:** CNN/DailyMail v3.0.0
- **Training Samples:** ~5,740 (1/50th of full dataset)
- **Validation Samples:** ~320 (1/40th of full dataset)
- **Test Samples:** ~280 (1/40th of full dataset)

## 📊 Performance Metrics

| Metric | Original Model | Fine-Tuned Model | Improvement |
|--------|---------------|------------------|-------------|
| ROUGE-1 | Baseline | +15.85% | ✅ |
| ROUGE-2 | Baseline | Improved | ✅ |
| ROUGE-L | Baseline | Improved | ✅ |

The fine-tuned model demonstrates significant improvements across all ROUGE metrics, indicating better overlap with human-written summaries in terms of unigrams, bigrams, and longest common subsequences.

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch==1.13.1 torchdata==0.5.1
pip install transformers==4.27.2 datasets==2.11.0 evaluate==0.4.0
pip install rouge_score==0.1.2 peft==0.3.0
pip install nltk
```

### Using the Pre-Trained Model

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("hardikJ11/bart-base-finetuned-cnn-news")
model = AutoModelForSeq2SeqLM.from_pretrained("hardikJ11/bart-base-finetuned-cnn-news")

# Generate summary
article = "Your news article text here..."
inputs = tokenizer(article, max_length=1024, return_tensors="pt", truncation=True)
outputs = model.generate(inputs["input_ids"], max_new_tokens=150)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(summary)
```

## 🔬 Technical Implementation

### 1. Data Preprocessing
- **Tokenization:** Custom tokenization pipeline handling both articles and summaries
- **Truncation Strategy:** Input articles truncated to 1024 tokens, summaries to 150 tokens
- **Data Augmentation:** Dataset shuffling and stratified sampling for efficient training

### 2. Training Configuration
```python
Hyperparameters:
- Batch Size: 8
- Epochs: 5
- Learning Rate: 5.6e-4
- Weight Decay: 0.01
- Optimization: AdamW (default in Seq2SeqTrainer)
- Evaluation Strategy: Per epoch
```

### 3. Fine-Tuning Approach
- **Method:** Full fine-tuning (all 139M parameters)
- **Training Framework:** Hugging Face Transformers + Seq2SeqTrainer
- **Data Collator:** DataCollatorForSeq2Seq with dynamic padding
- **Generation Strategy:** Beam search decoding

### 4. Evaluation Metrics
- **ROUGE-1:** Unigram overlap between generated and reference summaries
- **ROUGE-2:** Bigram overlap measuring fluency
- **ROUGE-L:** Longest common subsequence measuring coherence
- **Computation:** Uses stemming and sentence-level tokenization (NLTK)

## 📁 Project Structure

```
news-sum/
│
├── NewsSum.ipynb                 # Main notebook with complete pipeline
├── README.md                     # Project documentation
├── bart-base-finetuned-cnn-news/ # Training outputs and checkpoints
│   ├── checkpoint-*/             # Intermediate model checkpoints
│   └── runs/                     # TensorBoard logs
└── LICENSE
```

## 🔧 Methodology

### Training Pipeline
1. **Data Loading:** Load CNN/DailyMail dataset from Hugging Face
2. **Preprocessing:** Tokenize articles and summaries with appropriate length constraints
3. **Model Initialization:** Load pre-trained BART-base weights
4. **Fine-Tuning:** Train on news summarization task for 5 epochs
5. **Evaluation:** Compute ROUGE scores on held-out test set
6. **Deployment:** Push final model to Hugging Face Hub

### Inference Pipeline
1. **Input Processing:** Tokenize article with max length 1024
2. **Generation:** Use greedy decoding or beam search
3. **Post-Processing:** Decode tokens to text, remove special tokens
4. **Output:** Return abstractive summary (~150 tokens)

## 🎓 Key Learnings & Engineering Decisions

1. **Dataset Sampling:** Used 1/50th of training data to balance computational efficiency with model performance
2. **Model Selection:** BART chosen for strong seq2seq capabilities in text generation tasks
3. **Evaluation Framework:** ROUGE metrics provide interpretable, industry-standard assessment
4. **Model Versioning:** Implemented checkpoint saving and Hub integration for reproducibility
5. **GPU Optimization:** Explicit CUDA cache clearing to prevent OOM errors during training


## 📄 License

This project is licensed under the terms specified in the LICENSE file.

---

**Author:** Hardik Jain  
**Model:** Available on [Hugging Face Hub](https://huggingface.co/hardikJ11/bart-base-finetuned-cnn-news)  
**Task:** Abstractive Text Summarization  
**Domain:** Natural Language Processing (NLP) using Hugging Face Transformers
