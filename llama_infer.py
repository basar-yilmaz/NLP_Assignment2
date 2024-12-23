import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch
import json
import os
import argparse
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm


def setup_log_directory(dataset_type: str) -> str:
    """Create and return log directory path with current datetime."""
    log_dir = f"logs/{dataset_type}_llama"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"{timestamp}.log")


def setup_logging(dataset_type: str):
    """Configure logging settings with file output."""
    log_file = setup_log_directory(dataset_type)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Decoder Inference for Political Speech Classification"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Name of the decoder model to use",
    )
    parser.add_argument(
        "--text_type",
        type=str,
        choices=["text_en", "text"],
        default="text_en",
        help="Column to use for speech text (text_en or text)",
    )
    return parser.parse_args()


def setup_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Initialize and setup the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        attn_implementation="flash_attention_2",
        offload_folder="offload",
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


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


def load_validation_data(file_path: str, dataset_type: str) -> pd.DataFrame:
    """
    Load only validation data using cached split.
    We need this function to use consistent test data across different experiments and different models.
    """
    # Get validation IDs path
    val_ids_path = os.path.join("data", "splits", dataset_type, "val_ids.csv")

    if not os.path.exists(val_ids_path):
        raise FileNotFoundError(f"Validation IDs file not found: {val_ids_path}")

    # Load validation IDs
    val_ids = pd.read_csv(val_ids_path)["id"].tolist()

    # Load full dataset
    df = pd.read_csv(file_path, sep="\t")

    # Filter dataset to include only validation rows
    df_val = df[df["id"].isin(val_ids)]
    return df_val


def build_system_prompt_for_orientation_classification(speech: str) -> str:
    """Generate the prompt for classification. This is used only for orientation classification."""
    system_prompt = """
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant that is an expert in detecting the sentiment of a speech.
          Given a parliamentary speech in one of several languages, identify the ideology of the speaker's party. 
          In other words, this involves performing binary classification to determine whether the speaker's party leans left (0) or right (1).

          Your task is to give the classification in the given structure:
          {
            "speech": the classification of the speech,
            "confidence": the confidence of the classification
          }

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        WRITE YOUR ANSWER IN THE FORM OF GIVEN STRUCTURE, DO NOT WRITE ADDITIONAL EXPLANATIONS.
        """
    return f"{system_prompt}\n{speech}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|"


def build_system_prompt_for_governing_classification(speech: str) -> str:
    """Generate the prompt for classification. This is used only for governing (power) classification."""
    system_prompt = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant that is an expert in detecting the sentiment of a speech.
    Given a parliamentary speech in one of several languages, identify whether the speaker’s party is currently governing (0) or in opposition (1).

    Your task is to give the classification in the given structure:
    {
      "speech": the classification of the speech,
      "confidence": the confidence of the classification
    }

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    WRITE YOUR ANSWER IN THE FORM OF GIVEN STRUCTURE, DO NOT WRITE ADDITIONAL EXPLANATIONS.
    """
    return f"{system_prompt}\n{speech}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|"


def generate_prediction(
    classification_type: str,
    speech: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> Optional[Dict]:
    """Generate prediction for a single speech."""
    if classification_type == "orientation":
        prompt = build_system_prompt_for_orientation_classification(speech)
    elif classification_type == "power":
        prompt = build_system_prompt_for_governing_classification(speech)
    else:
        raise ValueError("Invalid classification type")

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        encoded_prompt = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_token_len = encoded_prompt.input_ids.shape[-1] + 1

        generated_ids = model.generate(
            **encoded_prompt,
            max_new_tokens=4096,
            pad_token_id=tokenizer.eos_token_id,
        )

        decoded_output = tokenizer.decode(
            generated_ids[0][input_token_len:],
            skip_special_tokens=True,
            return_full_text=False,
        )

    try:
        return json.loads(decoded_output)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing output: {e}\nOutput: {decoded_output}")
        return None


def process_speeches(
    dataset_type: str,
    df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text_type: str = "text_en",
) -> pd.DataFrame:
    """Process all speeches in the dataset."""
    if text_type not in df.columns:
        raise ValueError(f"Column {text_type} not found in dataset")

    results = []
    progress_bar = tqdm(
        df.iterrows(), total=len(df), desc="Processing speeches", unit="speech"
    )

    for idx, row in progress_bar:
        prediction = generate_prediction(
            classification_type=dataset_type,
            speech=row[text_type],
            model=model,
            tokenizer=tokenizer,
        )

        results.append(
            {
                "id": row["id"],
                "classification": prediction.get("speech") if prediction else None,
                "confidence": prediction.get("confidence") if prediction else None,
            }
        )

        # Update progress bar postfix with current speech ID
        progress_bar.set_postfix(speech_id=row["id"])

    results_df = pd.DataFrame(results)
    return df.merge(results_df, on="id")


def evaluate_results(df: pd.DataFrame) -> float:
    """Evaluate the classification results."""
    valid_rows = df.dropna(subset=["classification", "label"])
    accuracy = (valid_rows["classification"] == valid_rows["label"]).mean()

    logging.info(f"Accuracy: {accuracy:.2%}")
    logging.info(f"Valid predictions: {len(valid_rows)}/{len(df)}")

    return accuracy


def main():
    # Setup
    args = parse_args()
    dataset_type = get_dataset_type(args.data_path)
    setup_logging(dataset_type)

    # Load validation data
    df = load_validation_data(args.data_path, dataset_type)

    logging.info(f"Loaded {len(df)} validation samples")
    logging.info(f"Using text column: {args.text_type}")

    # Setup model
    model, tokenizer = setup_model(args.model_name)

    # Process speeches
    df_results = process_speeches(
        dataset_type, df, model, tokenizer, text_type=args.text_type
    )

    # Evaluate results
    accuracy = evaluate_results(df_results)

    # Print results
    logging.info(f"Accuracy: {accuracy:.4%}")

    # # Save results
    # output_path = f"results_{dataset_type}_validation.csv"
    # df_results.to_csv(output_path, index=False)
    # logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
