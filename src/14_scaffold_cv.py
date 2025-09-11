# 파일 경로: src/14_scaffold_cv.py (완전판)

import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import xgboost as xgb
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
# gat_regression_v4.py가 같은 src 폴더에 있다고 가정합니다.
from gat_regression_v4 import GAT, MoleculeDataset 

from sklearn.model_selection import KFold, cross_validate

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
GAT_MODEL_PATH = "models/gat_v4_model.pt"
CHEMBERTA_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
TARGET_COLUMN = 'label'
# --- 설정 끝 ---

# 특징 추출 함수들 (전체 코드 포함)
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


# 스캐폴드 기반 교차 검증 분할 함수
def scaffold_cv_split(df, n_splits=5, shuffle=True, random_state=42):
    print("\n스캐폴드 기반으로 데이터를 분할합니다...")
    scaffolds = defaultdict(list)
    for i, smiles in enumerate(df['smiles']):
        try:
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
        except:
            scaffold_smiles = smiles
        scaffolds[scaffold_smiles].append(i)

    unique_scaffolds = list(scaffolds.keys())
    if len(unique_scaffolds) < n_splits:
        raise ValueError(f"분할 수({n_splits})가 고유 스캐폴드 개수({len(unique_scaffolds)})보다 많습니다.")
        
    scaffold_kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    for train_scaffold_idx, test_scaffold_idx in scaffold_kfold.split(unique_scaffolds):
        train_idx = [idx for i in train_scaffold_idx for idx in scaffolds[unique_scaffolds[i]]]
        test_idx = [idx for i in test_scaffold_idx for idx in scaffolds[unique_scaffolds[i]]]
        yield np.array(train_idx), np.array(test_idx)


# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # 1. 데이터 및 모델 로드
    df = pd.read_csv(DATA_FILE_PATH)
    df.columns = [col.lower() for col in df.columns]
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol', TARGET_COLUMN])

    gat_model = GAT(num_node_features=17).to(DEVICE)
    gat_model.load_state_dict(torch.load(GAT_MODEL_PATH, map_location=DEVICE))
    chemberta_model = AutoModel.from_pretrained(CHEMBERTA_CHECKPOINT).to(DEVICE)
    chemberta_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_CHECKPOINT)
    
    # 2. 특징 추출
    print("\n--- 특징 추출 시작 ---")
    gat_dataset = MoleculeDataset(df, TARGET_COLUMN)
    gat_loader = DataLoader(gat_dataset, batch_size=128, shuffle=False)
    gat_features = get_gat_embeddings(gat_loader, gat_model)
    chemberta_features = get_chemberta_embeddings(df['smiles'].tolist(), chemberta_model, chemberta_tokenizer)
    X = np.concatenate([gat_features, chemberta_features], axis=1)
    y = df[TARGET_COLUMN].values

    # 3. 모델 정의 및 교차 검증 실행
    final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, 
                                   learning_rate=0.05, random_state=42)
    
    cv_split = scaffold_cv_split(df, n_splits=5, random_state=42)
    
    print("\n--- 스캐폴드 기반 5-Fold 교차 검증 시작 ---")
    cv_results = cross_validate(final_model, X, y, cv=cv_split, 
                                scoring=('r2', 'neg_mean_squared_error'), n_jobs=-1, verbose=1)
    
    r2_scores = cv_results['test_r2']
    mse_scores = -cv_results['test_neg_mean_squared_error']

    # 4. 최종 결과 출력
    print("\n" + "="*50)
    print("      최종 모델 스캐폴드 기반 교차 검증 성능")
    print("="*50)
    print(f"각 Fold의 R-squared: {np.round(r2_scores, 4)}")
    print(f"-> 평균 R-squared: {np.mean(r2_scores):.4f} (표준편차: {np.std(r2_scores):.4f})\n")
    print(f"각 Fold의 MSE: {np.round(mse_scores, 4)}")
    print(f"-> 평균 MSE: {np.mean(mse_scores):.4f} (표준편차: {np.std(mse_scores):.4f})")
    print("="*50)