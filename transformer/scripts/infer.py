import logging
import sys
import yaml
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import numpy as np

logger = logging.getLogger(__name__)

def load_params(config_path):
    try:
        with open(config_path, 'r', encoding="UTF8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

def main():
    # Load parameters from YAML
    params = load_params('../config/nli_transformers.yaml')

    # Load model, tokenizer, and config
    model_path = params.get('model_path', './results')  # Default to './results'
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    # Labels
    label_mapping = config.id2label

    # Start inference loop
    print("Type 'exit' to quit the inference loop.")
    while True:
        if params['task'] == "xnli" or params['task'] == "mnli":
            premise = input("Enter premise (or type 'exit' to quit): ")
            if premise.lower() == "exit":
                break
            hypothesis = input("Enter hypothesis: ")
            inputs = tokenizer(premise, hypothesis, truncation=True, padding=True, return_tensors="pt").to(device)
        elif params['task'] == "qnli":
            question = input("Enter question (or type 'exit' to quit): ")
            if question.lower() == "exit":
                break
            sentence = input("Enter sentence: ")
            inputs = tokenizer(question, sentence, truncation=True, padding=True, return_tensors="pt")
        else:
            logger.error(f"Unsupported task: {params['task']}")
            sys.exit(1)

        # Run model inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        # Convert prediction to label
        predicted_label = label_mapping[str(predictions.item())]
        print(f"Predicted label: {predicted_label}")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    main()
