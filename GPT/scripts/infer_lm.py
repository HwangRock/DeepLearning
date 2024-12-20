import torch
import yaml
import sys
import os

from transformers import pipeline

def main():
    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/lm_gpt.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    timestamp = "1698714028"
    checkpoint = 'checkpoint-3200'
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp, checkpoint)))

    gpt2_infer = pipeline('text-generation', model=params['model'], tokenizer=params['model'])
    amazon_gpt2_infer = pipeline('text-generation', model=out_dir, tokenizer=params['model'])
    print(gpt2_infer("This book was amazing"))
    print(amazon_gpt2_infer("This book was amazing"))

if __name__ == "__main__":
    main()