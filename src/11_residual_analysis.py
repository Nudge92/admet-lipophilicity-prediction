# 파일 경로: src/11_residual_analysis.py

import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import xgboost as xgb
from tqdm import tqdm
from rdkit import Chem

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
CHEMBERTA_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
TARGET_COLUMN = 'label'
# --- 설정 끝 ---

# 특징 추출 함수 (이전과 동일)
def get_chemberta_embeddings(smiles_list, model, tokenizer):
    print("ChemBERTa 특징 추출 중...")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for smiles in tqdm(smiles_list):
            inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            cls_embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
    return np.concatenate(embeddings, axis=0)

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # 1. 데이터 준비
    df = pd.read_csv(DATA_FILE_PATH)
    df.columns = [col.lower() for col in df.columns]
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol', TARGET_COLUMN])

    # 2. ChemBERTa 특징 추출
    chemberta_model = AutoModel.from_pretrained(CHEMBERTA_CHECKPOINT).to(DEVICE)
    chemberta_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_CHECKPOINT)
    X = get_chemberta_embeddings(df['smiles'].tolist(), chemberta_model, chemberta_tokenizer)
    y = df[TARGET_COLUMN].values

    # 3. 훈련/테스트 데이터 분할 (random_state=42로 고정하여 항상 동일하게 분할)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 모델 훈련
    print("\nChemBERTa+XGBoost 모델을 훈련합니다...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, 
                             learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    # 5. 테스트 데이터 예측
    y_pred = model.predict(X_test)

    # 6. 잔차(Residuals) 계산
    residuals = y_test - y_pred

    # 7. 잔차도(Residual Plot) 시각화
    print("잔차도를 생성합니다...")
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Values (예측값)", fontsize=14)
    plt.ylabel("Residuals (실제값 - 예측값)", fontsize=14)
    plt.title("Residual Plot for ChemBERTa+XGBoost Model", fontsize=16)
    plt.grid(True)
    plt.savefig("residual_plot.png") # 그래프를 파일로 저장
    print("residual_plot.png 파일이 저장되었습니다.")
    plt.show()

    # 8. 오차가 큰 상위 5개 분자 정보 출력
    df_test = df.iloc[X_test.index] # 테스트셋에 해당하는 원본 데이터프레임
    df_test['predicted'] = y_pred
    df_test['residual'] = residuals
    df_test['abs_residual'] = np.abs(residuals)

    print("\n--- 예측 오차가 가장 큰 상위 5개 분자 ---")
    print(df_test.sort_values(by='abs_residual', ascending=False).head(5)[['smiles', TARGET_COLUMN, 'predicted', 'residual']])