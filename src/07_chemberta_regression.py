# 파일 경로: src/07_chemberta_regression.py (버전 문제 해결 최종본)

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# --- 설정 부분 ---
MODEL_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
# --- 설정 끝 ---

# 1. 데이터 로드 및 전처리
print("데이터를 로드합니다...")
df = pd.read_csv(DATA_FILE_PATH)
df.columns = [col.lower() for col in df.columns]
target_column = 'label' if 'label' in df.columns else 'exp'

if target_column != 'label':
    df = df.rename(columns={target_column: 'label'})
    
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# 2. 토크나이저 로딩 및 데이터 토큰화
print("토크나이저를 로드하고 데이터를 토큰화합니다...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    return tokenizer(examples["smiles"], truncation=True, padding="max_length", max_length=128)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# 3. 모델 로딩
print("사전 학습된 ChemBERTa 모델을 로드합니다...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=1)

# 4. 성능 지표 계산 함수 정의
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.squeeze()
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {"mse": mse, "r2": r2}

# =================================================================
# ★★★ 버전 문제 해결을 위한 '안전 모드' 설정 ★★★
# 5. TrainingArguments를 가장 단순한 형태로 수정합니다.
training_args = TrainingArguments(
    output_dir="./chemberta-lipo-finetuned", # 결과 저장 폴더
    num_train_epochs=10,                      # 훈련 횟수
    per_device_train_batch_size=32,         # 훈련 배치 사이즈
    per_device_eval_batch_size=128,         # 평가 배치 사이즈
    weight_decay=0.01,                      # 가중치 감소
    logging_steps=100,                      # 100 스텝마다 로그 기록
    # evaluation_strategy 등 버전 타는 인자 모두 삭제
)
# =================================================================

# 6. Trainer 객체 생성 및 훈련 시작
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset, # 평가는 훈련이 끝난 후 수동으로 진행
    compute_metrics=compute_metrics,
)

print("ChemBERTa 모델 파인튜닝을 시작합니다...")
trainer.train()

# 7. 훈련이 끝난 후, 최종 평가를 수동으로 실행
print("\n훈련 완료! 최종 성능을 평가합니다...")
final_results = trainer.evaluate()

print("\n" + "="*40)
print("     ChemBERTa 모델 최종 성능")
print("="*40)
print(f"평균 제곱 오차 (MSE): {final_results['eval_mse']:.4f}")
print(f"결정 계수 (R-squared): {final_results['eval_r2']:.4f}")
print("="*40)