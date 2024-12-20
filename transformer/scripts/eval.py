import logging
import sys
import os
import yaml
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import datasets
from datasets import load_dataset
import evaluate

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

    # Task-specific dataset loading
    valid_tasks = ["xnli", "qnli", "mnli"]
    if params['task'] not in valid_tasks:
        logger.error(f"Invalid task specified: {params['task']}. Supported tasks: {valid_tasks}")
        sys.exit(1)

    # Load evaluation dataset
    if params['task'] == "xnli":
        eval_dataset = datasets.load_dataset("xnli", "en", split="validation")
    elif params['task'] == "qnli":
        eval_dataset = datasets.load_dataset("glue", "qnli", split="validation")
    elif params['task'] == "mnli":
        eval_dataset = datasets.load_dataset("glue", "mnli", split="validation_matched")

    # Load model, tokenizer, and config
    model_path = params.get('model_path', './results')  # Default to './results'
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

    # Preprocessing function
    def preprocess_function(examples):
        return tokenizer(
            examples["premise"] if params['task'] != 'qnli' else examples["question"],
            examples["hypothesis"] if params['task'] != 'qnli' else examples["sentence"],
            truncation=True,
            max_length=params.get("max_seq_length", 128),
            padding="max_length",
        )

    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # Define accuracy metric
    metric = evaluate.load("accuracy")

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        return metric.compute(predictions=preds, references=pred.label_ids)

    # Load TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=params.get("eval_batch_size", 16),
        do_train=False,
        do_eval=True,
        logging_dir="./logs",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Evaluate
    logger.info("Starting evaluation...")
    results = trainer.evaluate()
    logger.info(f"Results: {results}")

    # Print accuracy
    print(f"Accuracy: {results['eval_accuracy']:.4f}")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    main()
