# 파일 경로: src/12_final_residual_analysis.py

import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import xgboost as xgb
from tqdm import tqdm
from rdkit import Chem

# torch_geometric 관련 임포트
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool

# 별도 파일로 분리한 GAT 모델 및 데이터셋 클래스 임포트
from gat_regression_v4 import GAT, MoleculeDataset

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
GAT_MODEL_PATH = "models/gat_v4_model.pt" # 훈련된 GAT 모델 경로
CHEMBERTA_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
TARGET_COLUMN = 'label'
# --- 설정 끝 ---

# 특징 추출 함수들 (수정된 최종 버전)
def get_gat_embeddings(loader, model):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in tqdm(loader, desc="GAT 특징 추출 중"):
            data = data.to(DEVICE)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for i in range(len(model.convs)):
                x = model.convs[i](x, edge_index)
                x = model.batch_norms[i](x)
                x = F.elu(x)
            graph_embedding = global_mean_pool(x, batch)
            embeddings.append(graph_embedding.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def get_chemberta_embeddings(smiles_list, model, tokenizer):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for smiles in tqdm(smiles_list, desc="ChemBERTa 특징 추출 중"):
            inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            cls_embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
    return np.concatenate(embeddings, axis=0)

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # 1. 데이터 및 모델 로드
    df = pd.read_csv(DATA_FILE_PATH)
    df.columns = [col.lower() for col in df.columns]
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol', TARGET_COLUMN])

    gat_model = GAT(num_node_features=17).to(DEVICE) # num_node_features는 이전에 확인한 값으로 설정
    gat_model.load_state_dict(torch.load(GAT_MODEL_PATH, map_location=DEVICE))
    chemberta_model = AutoModel.from_pretrained(CHEMBERTA_CHECKPOINT).to(DEVICE)
    chemberta_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_CHECKPOINT)

    # 2. 특징 추출
    print("\n--- 특징 추출 시작 ---")
    gat_dataset = MoleculeDataset(df, TARGET_COLUMN)
    gat_loader = DataLoader(gat_dataset, batch_size=128, shuffle=False)
    gat_features = get_gat_embeddings(gat_loader, gat_model)
    chemberta_features = get_chemberta_embeddings(df['smiles'].tolist(), chemberta_model, chemberta_tokenizer)
    
    # 하이브리드 특징 생성
    X = np.concatenate([gat_features, chemberta_features], axis=1)
    y = df[TARGET_COLUMN].values
    
    # 3. 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42
    )

    # 4. 모델 훈련
    print("\n최종 하이브리드 모델을 훈련합니다...")
    final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, 
                                   learning_rate=0.05, random_state=42)
    final_model.fit(X_train, y_train)

    # 5. 테스트 데이터 예측 및 잔차 계산
    y_pred = final_model.predict(X_test)
    residuals = y_test - y_pred

    # 6. 잔차도 시각화
    print("잔차도를 생성합니다...")
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Values (예측값)", fontsize=14)
    plt.ylabel("Residuals (실제값 - 예측값)", fontsize=14)
    plt.title("Residual Plot for Final Hybrid Model", fontsize=16)
    plt.grid(True)
    plt.savefig("final_residual_plot.png")
    print("final_residual_plot.png 파일이 저장되었습니다.")
    plt.show()

    # 7. 오차가 큰 상위 5개 분자 정보 출력
    df_test_output = df_test.copy()
    df_test_output['predicted'] = y_pred
    df_test_output['residual'] = residuals
    df_test_output['abs_residual'] = np.abs(residuals)

    print("\n--- 예측 오차가 가장 큰 상위 5개 분자 ---")
    print(df_test_output.sort_values(by='abs_residual', ascending=False).head(5)[['smiles', TARGET_COLUMN, 'predicted', 'residual']])