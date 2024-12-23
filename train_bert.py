import os
import argparse
import pandas as pd
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm



@dataclass
class Config:
    model_name: str = (
        "bert-base-multilingual-cased"  # pre-trained model name from Hugging Face
    )
    batch_size: int = 64  # batch size for training and validation
    max_length: int = 512  # since we are using BERT I will use max_length of 512
    epochs: int = 20  # number of epochs (we generally get early stopping)
    learning_rate: float = 2e-5  # init learning rate
    warmup_ratio: float = 0.1  # warmup steps ratio
    patience: int = 3  # early stopping
    val_size: float = 0.1  # validation set size (0.1 = 10%)
    dataset_type: str = "orientation"  # or "power"


def get_dataset_type(file_path: str) -> str:
    """Detect dataset type from file path."""
    file_name = os.path.basename(file_path).lower()
    if "orientation" in file_name:
        return "orientation"
    elif "power" in file_name:
        return "power"
    raise ValueError(
        "Dataset type not recognized. File name must contain 'orientation' or 'power'"
    )


def setup_output_dirs(dataset_type: str) -> tuple[str, str]:
    """Create and return paths for logs and saved models."""
    log_dir = os.path.join("logs", dataset_type)
    save_dir = os.path.join("saved", dataset_type)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    return log_dir, save_dir


def setup_logging(model_name: str, log_dir: str) -> str:
    """Setup logging with dataset-specific directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(
        log_dir, f"training_{timestamp}_{model_name.replace('/', '_')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )
    return log_filename


class ParliamentaryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text classification model")
    parser.add_argument("--data_path", required=True, help="Path to the TSV data file")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--text_type",
        choices=["text", "text_en"],
        required=True,
        help="Type of text to use (text or text_en)",
    )
    parser.add_argument(
        "--output_dir", default="saved", help="Base directory to save models"
    )
    return parser.parse_args()


def setup_device(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    if gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"GPU ID {gpu_id} is not available. Max GPU ID is {torch.cuda.device_count()-1}"
        )

    return torch.device(f"cuda:{gpu_id}")


def get_split_cache_paths(dataset_type: str) -> Tuple[str, str]:
    """Get paths for cached train/val splits."""
    cache_dir = os.path.join("data", "splits", dataset_type)
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "train_ids.csv")
    val_path = os.path.join(cache_dir, "val_ids.csv")
    return train_path, val_path

def save_split_indices(train_data: pd.DataFrame, val_data: pd.DataFrame, dataset_type: str):
    """Save train/val splits ids to csv files."""
    train_path, val_path = get_split_cache_paths(dataset_type)
    
    train_data[['id']].to_csv(train_path, index=False)
    val_data[['id']].to_csv(val_path, index=False)
    logging.info(f"Saved split indices to {train_path} and {val_path}")

def load_cached_splits(file_path: str, dataset_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/val splits from cached files."""
    train_path, val_path = get_split_cache_paths(dataset_type)
    
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return None, None
        
    # Load the full dataset
    data = pd.read_csv(file_path, sep="\t")
    
    # Load cached splits
    train_ids = pd.read_csv(train_path)['id']
    val_ids = pd.read_csv(val_path)['id']
    
    # Split the data using cached indices
    train_data = data[data['id'].isin(train_ids)]
    val_data = data[data['id'].isin(val_ids)]
    
    logging.info(f"Loaded cached splits from {train_path} and {val_path}")
    return train_data, val_data

def load_and_split_data(
    file_path: str, text_type: str, val_size: float, dataset_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Try to load cached splits first
    train_data, val_data = load_cached_splits(file_path, dataset_type)
    
    if train_data is None or val_data is None:
        # If no cache exists, create new split
        data = pd.read_csv(file_path, sep="\t")
        if "label" not in data.columns or text_type not in data.columns:
            raise ValueError(f"Required columns (label, {text_type}) not found in dataset")

        train_data, val_data = train_test_split(
            data, test_size=val_size, stratify=data["label"], random_state=42
        )
        
        # Save the splits for future use
        save_split_indices(train_data, val_data, dataset_type)
        
    return train_data, val_data


def create_data_loaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    text_type: str,
    tokenizer,
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = ParliamentaryDataset(
        train_data[text_type].tolist(),
        train_data["label"].tolist(),
        tokenizer,
        config.max_length,
    )
    val_dataset = ParliamentaryDataset(
        val_data[text_type].tolist(),
        val_data["label"].tolist(),
        tokenizer,
        config.max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(val_loader), correct / total


def main():
    args = parse_args()

    # Detect dataset type from file path
    dataset_type = get_dataset_type(args.data_path)

    # Initialize config with detected dataset type
    config = Config()
    config.dataset_type = dataset_type

    # Setup directory structure
    log_dir, save_dir = setup_output_dirs(dataset_type)

    # Setup logging
    log_filename = setup_logging(config.model_name, log_dir)
    device = setup_device(args.gpu_id)

    logging.info(f"Log file: {log_filename}")
    logging.info(f"Dataset type: {dataset_type}")
    logging.info(f"Starting training with configuration: {config}")
    logging.info(f"Using device: {device}")

    # Load and prepare data
    train_data, val_data = load_and_split_data(
        args.data_path, args.text_type, config.val_size, dataset_type
    )

    logging.info(f"Text type: {args.text_type}")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    ).to(device)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, args.text_type, tokenizer, config
    )

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    # Training loop
    best_val_loss = float("inf")
    wait = 0
    model_save_dir = os.path.join(save_dir, f"{config.model_name}_{args.text_type}")
    os.makedirs(model_save_dir, exist_ok=True)

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_accuracy = validate(model, val_loader, device)

        logging.info(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)
            logging.info(
                f"Better model saved to {model_save_dir} (val_loss: {val_loss:.4f})"
            )
        else:
            wait += 1
            if wait >= config.patience:
                logging.info("Early stopping triggered")
                break


if __name__ == "__main__":
    main()
