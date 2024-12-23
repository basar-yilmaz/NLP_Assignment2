# Parliamentary Debate Classification: Fine-Tuning and Zero-Shot Inference

This repository contains the source code for binary classification tasks on parliamentary speeches, focusing on **political orientation** and **governing power**. It includes implementations for fine-tuning a **masked language model (BERT)** and performing zero-shot inference with a **causal language model (LLaMA)**.

## Tasks

1. **Political Orientation**: Classify speeches as either left (0) or right (1).
2. **Governing Power**: Determine whether a speaker's party is part of the government (0) or in opposition (1).

Both tasks use the ParlaMint dataset, focusing on the Turkish language for binary classification.

---

## Repository Structure

```
├── llama_infer.py         # Script for zero-shot inference using a causal language model
├── train_bert.py          # Script for fine-tuning a masked language model
├── data/                  # Directory for datasets and splits
├── logs/                  # Directory for logs
├── saved/                 # Directory for saving fine-tuned models
└── README.md              # Project documentation
```

---

## Dataset

### ParlaMint Collection
The dataset contains parliamentary debates from 29 countries, including:
- `text`: Original Turkish speeches.
- `text_en`: English translations of speeches.
- `label`: Binary classification labels for political orientation and governing power.

### Preprocessing
1. Perform a stratified 90-10 train-test split.
2. Cache train and validation split indices in `data/splits/<task_type>` for consistent experimentation.

---

## Models

### Masked Language Model (Fine-Tuning)
- **Model**: `bert-base-multilingual-cased`
- **Use Case**: Fine-tuning for political orientation and governing power classification.
- **Implementation**: `train_bert.py`

### Causal Language Model (Zero-Shot)
- **Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Use Case**: Zero-shot inference for political orientation and governing power tasks.
- **Implementation**: `llama_infer.py`

---

## Installation and Setup

### Requirements
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Additional libraries: `pandas`, `tqdm`, `scikit-learn`

### Installation
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Fine-Tuning BERT

**Command:**
```bash
python train_bert.py --data_path <path_to_tsv_file> --text_type <text/text_en> --gpu_id <gpu_id>
```

**Arguments:**
- `--data_path`: Path to the dataset file (TSV format).
- `--text_type`: Type of text to use (`text` or `text_en`).
- `--gpu_id`: GPU ID to use for training (default: 0).

**Example:**
```bash
python train_bert.py --data_path data/turkish_orient.tsv --text_type text --gpu_id 0
```

**Outputs:**
- Fine-tuned model saved in `saved/<task_type>`.
- Training logs stored in `logs/<task_type>`.

---

### 2. Zero-Shot Inference with LLaMA

**Command:**
```bash
python llama_infer.py --data_path <path_to_tsv_file> --model_name <model_name> --text_type <text/text_en>
```

**Arguments:**
- `--data_path`: Path to the dataset file (TSV format).
- `--model_name`: Pretrained causal language model (default: `meta-llama/Llama-3.1-8B-Instruct`).
- `--text_type`: Type of text to use (`text` or `text_en`).

**Example:**
```bash
python llama_infer.py --data_path data/turkish_power.tsv --text_type text_en
```

**Outputs:**
- Classification accuracy printed in logs.
- Logs stored in `logs/<task_type>`.

---

## Results

### Political Orientation
| Model                  | Text Type | Accuracy  |
|------------------------|-----------|-----------|
| BERT (Fine-Tuned)      | `text`    | 86.07%    |
| BERT (Fine-Tuned)      | `text_en` | 88.32%    |
| LLaMA (Zero-Shot)      | `text`    | 73.79%    |
| LLaMA (Zero-Shot)      | `text_en` | 72.42%    |

### Governing Power
| Model                  | Text Type | Accuracy  |
|------------------------|-----------|-----------|
| BERT (Fine-Tuned)      | `text`    | 86.37%    |
| BERT (Fine-Tuned)      | `text_en` | 86.72%    |
| LLaMA (Zero-Shot)      | `text`    | 73.71%    |
| LLaMA (Zero-Shot)      | `text_en` | 69.53%    |

---

## Limitations and Future Work

### Limitations
- **Class Imbalance**: Slight label skewness may affect model performance.
- **Token Length Constraints**: Loss of context in lengthy speeches due to maximum token length restrictions in BERT.
- **Cross-Lingual Generalization**: LLaMA struggles with zero-shot inference on English-translated texts.

### Future Work
- **Data Augmentation**: Introduce techniques like back-translation to enhance robustness.
- **Fine-Tuning LLaMA**: Explore task-specific fine-tuning of causal language models.
- **Ensemble Learning**: Combine BERT and LLaMA predictions for improved accuracy.

---


