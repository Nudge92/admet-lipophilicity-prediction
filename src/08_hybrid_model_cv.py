# 파일 경로: src/08_hybrid_model_cv.py (교차 검증 버전)

import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F # F.elu를 사용하기 위해 추가
import xgboost as xgb
from tqdm import tqdm
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel

# torch_geometric 관련 임포트
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool # global_mean_pool을 사용하기 위해 추가

# 별도 파일로 분리한 GAT 모델 및 데이터셋 클래스 임포트
from gat_regression_v4 import GAT, MoleculeDataset

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
GAT_MODEL_PATH = "models/gat_v4_model.pt" # 훈련된 GAT 모델 경로
CHEMBERTA_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
# --- 설정 끝 ---

# GAT 모델 특징 추출 함수 수정
def get_gat_embeddings(loader, model):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in tqdm(loader, desc="GAT 특징 추출 중"):
            data = data.to(DEVICE)
            # GAT 모델의 forward 로직을 그대로 재현하여 최종 그래프 임베딩을 추출합니다.
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            for i in range(len(model.convs)):
                x = model.convs[i](x, edge_index)
                x = model.batch_norms[i](x)
                x = F.elu(x)
                # eval 모드이므로 dropout은 적용되지 않습니다.
            
            # global_mean_pool을 적용하여 그래프 단위 임베딩 생성
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

# --- 메인 실행 로직 (교차 검증) ---
if __name__ == '__main__':
    # 1. 전체 데이터 준비
    df = pd.read_csv(DATA_FILE_PATH)
    df.columns = [col.lower() for col in df.columns]
    target_column = 'label' if 'label' in df.columns else 'exp'
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol', target_column])
    print(f"전체 데이터셋 크기: {len(df)}")

    # 2. 특징 추출 (전체 데이터에 대해 1회만 수행)
    print("\n--- 특징 추출 시작 ---")
    
    # GAT 특징 추출
    gat_dataset = MoleculeDataset(df, target_column)
    gat_loader = DataLoader(gat_dataset, batch_size=128, shuffle=False)
    gat_model = GAT(num_node_features=gat_dataset.num_node_features).to(DEVICE)
    try:
        gat_model.load_state_dict(torch.load(GAT_MODEL_PATH, map_location=DEVICE))
        gat_features = get_gat_embeddings(gat_loader, gat_model)
    except FileNotFoundError:
        print(f"경고: GAT 모델 파일({GAT_MODEL_PATH})을 찾을 수 없습니다. GAT 특징을 제외합니다.")
        gat_features = None

    # ChemBERTa 특징 추출
    chemberta_model = AutoModel.from_pretrained(CHEMBERTA_CHECKPOINT).to(DEVICE)
    chemberta_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_CHECKPOINT)
    chemberta_features = get_chemberta_embeddings(df['smiles'].tolist(), chemberta_model, chemberta_tokenizer)
    
    # 하이브리드 특징 생성
    if gat_features is not None:
        print("\nGAT와 ChemBERTa 특징을 결합합니다.")
        hybrid_features = np.concatenate([gat_features, chemberta_features], axis=1)
    else:
        hybrid_features = chemberta_features

    X = hybrid_features
    y = df[target_column].values

    # 3. 5-Fold 교차 검증을 위한 최종 모델 및 CV 객체 정의
    from sklearn.model_selection import KFold, cross_val_score
    
    final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, 
                                   learning_rate=0.05, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 4. 교차 검증 실행
    print("\n--- 5-Fold 교차 검증 시작 ---")
    r2_scores = cross_val_score(final_model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    mse_scores = cross_val_score(final_model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1) * -1
    
    # 5. 최종 결과 출력
    print("\n" + "="*40)
    print("      하이브리드 모델 교차 검증 최종 성능")
    print("="*40)
    print(f"각 Fold의 R-squared: {np.round(r2_scores, 4)}")
    print(f"-> 평균 R-squared: {np.mean(r2_scores):.4f} (표준편차: {np.std(r2_scores):.4f})\n")
    print(f"각 Fold의 MSE: {np.round(mse_scores, 4)}")
    print(f"-> 평균 MSE: {np.mean(mse_scores):.4f} (표준편차: {np.std(mse_scores):.4f})")
    print("="*40)