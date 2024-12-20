import time
import os
import logging
import sys
import yaml
import math

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
import datasets

logger = logging.getLogger(__name__)

def main():

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/lm_gpt.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # 랜덤 시드 세팅
    if 'random_seed' in params:
        set_seed(params['random_seed'])

    # 데이터 로드
    if params['task'] == "Book":  # huggingface datasets로 부터 bookcorpus load
        train_data = datasets.load_dataset("bookcorpus", split="train[:10000]")
        val_data = datasets.load_dataset("bookcorpus", split="train[-1000:]")
        print(len(train_data))
        print(len(val_data))
    elif params['task'] == "Amazon":  # custom dataset을 load
        train_file = params['data_files']['Amazon']['train_file']
        val_file = params['data_files']['Amazon']['val_file']

        # 파일에서 데이터를 읽어옴
        with open(train_file, 'r', encoding="UTF8") as f:
            train_texts = f.readlines()

        with open(val_file, 'r', encoding="UTF8") as f:
            val_texts = f.readlines()

        # 데이터를 datasets.Dataset 형태로 변환
        train_data = datasets.Dataset.from_dict({"text": train_texts})
        val_data = datasets.Dataset.from_dict({"text": val_texts})
        print(len(train_data))
        print(len(val_data))

    # 모델 및 토크나이저 설정
    model = AutoModelForCausalLM.from_pretrained("gpt2-large")  # GPT-2 Large 모델 로드
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")    # GPT-2 Large 토크나이저 로드
    tokenizer.pad_token = tokenizer.eos_token                 # PAD 토큰 설정

    if params['max_seq_length'] is None:
        params['max_seq_length'] = tokenizer.model_max_length
        if params['max_seq_length'] > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            params['max_seq_length'] = 1024
    else:
        if params['max_seq_length'] > tokenizer.model_max_length:
            logger.warn(
                f"The max_seq_length passed ({params['max_seq_length']}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        params['max_seq_length'] = min(params['max_seq_length'], tokenizer.model_max_length)

    print(params['max_seq_length'])

    def process_data_to_model_inputs(batch):
        inputs = tokenizer(batch["text"], padding='max_length', truncation=True, max_length=params['max_seq_length'])
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        return batch

    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=params['batch_size'],
        remove_columns=["text"]
    )
    train_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )
    val_data = val_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=params['batch_size'],
        remove_columns=["text"]
    )
    val_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )

    # 학습 설정
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))

    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        learning_rate=params['optimizer_params'][params['optimizer']]['lr'],
        num_train_epochs=params['max_epochs'],
        logging_dir=out_dir,
        logging_steps=params['logging_steps'],
        save_steps=params['save_steps'],
        eval_steps=params['eval_steps'],
        warmup_steps=params['warmup_steps'],
        overwrite_output_dir=True,
        save_total_limit=3,
        prediction_loss_only=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    trainer.train()
    trainer.save_model()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

if __name__ == '__main__':
    main()
