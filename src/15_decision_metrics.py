# 파일 경로: src/15_decision_metrics.py (완전판)

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
from gat_regression_v4 import GAT, MoleculeDataset

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
GAT_MODEL_PATH = "models/gat_v4_model.pt"
CHEMBERTA_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
TARGET_COLUMN = 'label'
HIT_QUANTILE = 0.8 # 상위 20%를 'Hit'으로 정의
EF_PERCENTAGE = 0.05 # 상위 5%에서의 농축 계수 계산
# --- 설정 끝 ---

# --- 함수 정의 부분 (전체 포함) ---

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

def scaffold_cv_split(df, n_splits=5, shuffle=True, random_state=42):
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

def calculate_enrichment_factor(y_true, y_pred, percentage):
    sorted_indices = np.argsort(y_pred)[::-1]
    top_n_indices = sorted_indices[:int(len(y_pred) * percentage)]
    hit_rate_total = np.sum(y_true) / len(y_true)
    hit_rate_top_n = np.sum(y_true[top_n_indices]) / len(top_n_indices)
    if hit_rate_total == 0: return float('inf')
    return hit_rate_top_n / hit_rate_total

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # 1. 데이터 및 특징 준비
    df = pd.read_csv(DATA_FILE_PATH)
    df.columns = [col.lower() for col in df.columns]
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol', TARGET_COLUMN])
    gat_model = GAT(num_node_features=17).to(DEVICE)
    gat_model.load_state_dict(torch.load(GAT_MODEL_PATH, map_location=DEVICE))
    chemberta_model = AutoModel.from_pretrained(CHEMBERTA_CHECKPOINT).to(DEVICE)
    chemberta_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_CHECKPOINT)
    print("\n--- 특징 추출 시작 ---")
    gat_dataset = MoleculeDataset(df, TARGET_COLUMN)
    gat_loader = DataLoader(gat_dataset, batch_size=128, shuffle=False)
    gat_features = get_gat_embeddings(gat_loader, gat_model)
    chemberta_features = get_chemberta_embeddings(df['smiles'].tolist(), chemberta_model, chemberta_tokenizer)
    X = np.concatenate([gat_features, chemberta_features], axis=1)
    y = df[TARGET_COLUMN].values

    # 2. 단일 스캐폴드 분할로 훈련/테스트셋 생성
    print("\n스캐폴드 기반으로 데이터를 훈련/테스트셋으로 분할합니다...")
    train_idx, test_idx = next(scaffold_cv_split(df, n_splits=5, random_state=42))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 3. 모델 훈련 및 예측
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, 
                             learning_rate=0.05, random_state=42)
    print("\n모델을 훈련합니다...")
    model.fit(X_train, y_train)
    y_pred_reg = model.predict(X_test)

    # 4. 분류 문제로 변환 및 성능 평가
    hit_threshold = np.quantile(y, HIT_QUANTILE)
    y_true_class = (y_test >= hit_threshold).astype(int)
    roc_auc = roc_auc_score(y_true_class, y_pred_reg)
    ef_5 = calculate_enrichment_factor(y_true_class, y_pred_reg, percentage=EF_PERCENTAGE)

    # 5. 결과 출력
    print("\n" + "="*50)
    print("           의사결정 지표 기반 성능 평가")
    print("="*50)
    print(f"'Hit' 기준값 (상위 {100-HIT_QUANTILE*100:.0f}%): {hit_threshold:.4f}")
    print(f"테스트셋의 'Hit' 개수: {np.sum(y_true_class)} / {len(y_true_class)}")
    print("-" * 50)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Enrichment Factor @ {EF_PERCENTAGE*100:.0f}% (EF5): {ef_5:.2f}")
    print("="*50)

    # 6. ROC 커브 시각화
    fpr, tpr, _ = roc_curve(y_true_class, y_pred_reg)
    plt.figure(figsize=(9, 9))
    plt.plot(fpr, tpr, lw=2, label=f'Final Hybrid Model (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve for Final Hybrid Model', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("final_roc_curve.png")
    print("\nfinal_roc_curve.png 파일이 저장되었습니다.")
    plt.show()