import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import datasets


def main():
    # 학습된 모델 디렉토리
    model_dir = "./results"  # 학습된 모델이 저장된 디렉토리
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # 추론할 데이터
    test_data = [
        "John lives in New York.",
        "Apple is looking at buying U.K. startup for $1 billion.",
    ]  # 추론 예시 데이터

    # 데이터 전처리
    def preprocess_texts(texts):
        tokenized_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,  # 학습 시 max_seq_length와 동일하게 설정
            return_tensors="pt",
        )
        return tokenized_inputs

    # 텍스트 추론
    def predict(texts):
        tokenized_inputs = preprocess_texts(texts)
        input_ids = tokenized_inputs["input_ids"].to(device)
        attention_mask = tokenized_inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # 토큰별 라벨 디코딩
        decoded_predictions = []
        for i, pred in enumerate(predictions):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            labels = [model.config.id2label[label_id] for label_id in pred]
            decoded_predictions.append(list(zip(tokens, labels)))

        return decoded_predictions

    # 추론 실행
    results = predict(test_data)

    # 결과 출력
    for i, result in enumerate(results):
        print(f"\nInput {i + 1}: {test_data[i]}")
        print("Prediction:")
        for token, label in result:
            print(f"{token}: {label}")


if __name__ == "__main__":
    main()
