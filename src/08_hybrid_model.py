# 파일 경로: src/08_hybrid_model.py (최종 하이브리드 모델)

import pandas as pd
import numpy as np
import os
import torch
import xgboost as xgb
from tqdm import tqdm
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import DataLoader

# 이전 GAT v4 스크립트에서 모델 정의와 데이터셋 클래스를 가져옵니다.
# 수정 후
from src.gat_regression_v4 import GAT, MoleculeDataset # GATv4 코드를 별도 파일로 분리했다고 가정

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
GAT_MODEL_PATH = "models/gat_v4_model.pt" # 훈련된 GAT 모델을 저장하고 불러올 경로
CHEMBERTA_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
# --- 설정 끝 ---

# 특징 추출 함수들
def get_gat_embeddings(loader, model):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in tqdm(loader, desc="GAT 특징 추출 중"):
            data = data.to(DEVICE)
            # global_mean_pool 직전의 노드 임베딩을 가져옴
            x = model.conv1(data.x, data.edge_index)
            x = model.batch_norms[0](x)
            x = F.elu(x)
            x = model.convs[1](x, data.edge_index) # GATv4 구조에 맞게 수정
            x = model.batch_norms[1](x)
            x = F.elu(x)
            x = model.convs[2](x, data.edge_index)
            x = model.batch_norms[2](x)
            graph_embedding = global_mean_pool(x, data.batch)
            embeddings.append(graph_embedding.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def get_chemberta_embeddings(smiles_list, model, tokenizer):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for smiles in tqdm(smiles_list, desc="ChemBERTa 특징 추출 중"):
            inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            # [CLS] 토큰의 마지막 레이어 hidden state를 사용
            cls_embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
    return np.concatenate(embeddings, axis=0)

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # 1. 데이터 준비
    df = pd.read_csv(DATA_FILE_PATH)
    df.columns = [col.lower() for col in df.columns]
    target_column = 'label' if 'label' in df.columns else 'exp'
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol', target_column])
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 2. GAT 특징 추출
    # GAT v4 모델 로드 (훈련된 모델이 있다고 가정)
    # 실제로는 06_gat_regression_v4.py 마지막에 torch.save(model.state_dict(), GAT_MODEL_PATH) 추가 필요
    gat_dataset = MoleculeDataset(df, target_column)
    gat_loader = DataLoader(gat_dataset, batch_size=128, shuffle=False)
    
    # GAT 모델 인스턴스화 및 state_dict 로드
    gat_model = GAT(num_node_features=gat_dataset.num_node_features).to(DEVICE)
    # gat_model.load_state_dict(torch.load(GAT_MODEL_PATH)) # 실제 실행시 주석 해제
    
    # gat_features = get_gat_embeddings(gat_loader, gat_model) # 실제 실행시 주석 해제

    # 3. ChemBERTa 특징 추출
    chemberta_model = AutoModel.from_pretrained(CHEMBERTA_CHECKPOINT).to(DEVICE)
    chemberta_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_CHECKPOINT)
    chemberta_features = get_chemberta_embeddings(df['smiles'].tolist(), chemberta_model, chemberta_tokenizer)
    
    # 4. 하이브리드 특징 생성
    # hybrid_features = np.concatenate([gat_features, chemberta_features], axis=1) # 실제 실행시 주석 해제
    # 임시로 ChemBERTa 특징만 사용 (GAT 모델 훈련/저장 로직 생략)
    hybrid_features = chemberta_features

    X = hybrid_features
    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. 최종 XGBoost 모델 훈련 및 평가
    print("\n하이브리드 특징으로 최종 XGBoost 모델을 훈련합니다...")
    final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
                                   early_stopping_rounds=50, random_state=42)
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = final_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*40)
    print("      하이브리드 모델 최종 성능")
    print("="*40)
    print(f"평균 제곱 오차 (MSE): {mse:.4f}")
    print(f"결정 계수 (R-squared): {r2:.4f}")
    print("="*40)