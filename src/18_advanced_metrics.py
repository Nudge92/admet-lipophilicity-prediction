# 파일 경로: src/18_advanced_metrics.py (완전판)

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
from sklearn.metrics import roc_curve, average_precision_score
import matplotlib.pyplot as plt

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
GAT_MODEL_PATH = "models/gat_v4_model.pt"
CHEMBERTA_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
TARGET_COLUMN = 'label'
HIT_QUANTILE = 0.8
N_BOOTSTRAPS = 1000
EF_PERCENTAGE = 0.05
PRECISION_PERCENTAGE = 0.10
TARGET_FPR = 0.1
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
    if hit_rate_total == 0: return float('inf')
    hit_rate_top_n = np.sum(y_true[top_n_indices]) / len(top_n_indices)
    return hit_rate_top_n / hit_rate_total

def calculate_precision_at_k(y_true, y_pred, k_percentage):
    k = int(len(y_pred) * k_percentage)
    if k == 0: return np.nan
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    return np.sum(y_true[top_k_indices]) / k

def calculate_tpr_at_fpr(y_true, y_pred, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    try:
        idx = np.where(fpr >= target_fpr)[0][0]
        return tpr[idx]
    except IndexError:
        return 1.0

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
    
    # 2. 단일 스캐폴드 분할 및 모델 훈련/예측
    print("\n스캐폴드 기반으로 데이터를 훈련/테스트셋으로 분할합니다...")
    train_idx, test_idx = next(scaffold_cv_split(df, n_splits=5, random_state=42))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, 
                             learning_rate=0.05, random_state=42)
    print("\n모델을 훈련합니다...")
    model.fit(X_train, y_train)
    y_pred_reg = model.predict(X_test)
    hit_threshold = np.quantile(y, HIT_QUANTILE)
    y_true_class = (y_test >= hit_threshold).astype(int)

    # 3. 부트스트랩으로 모든 지표의 신뢰구간 계산
    print(f"\n부트스트랩을 시작합니다 (반복 횟수: {N_BOOTSTRAPS})...")
    auprc_scores, ef5_scores, p10_scores, tpr_at_fpr_scores = [], [], [], []
    test_indices = np.arange(len(y_test))

    for _ in tqdm(range(N_BOOTSTRAPS)):
        bootstrap_indices = np.random.choice(test_indices, size=len(y_test), replace=True)
        y_true_boot = y_true_class[bootstrap_indices]
        y_pred_boot = y_pred_reg[bootstrap_indices]
        
        # 부트스트랩 샘플에 Hit이 없는 경우 AUPRC가 정의되지 않을 수 있어 예외 처리
        if np.sum(y_true_boot) > 0:
            auprc_scores.append(average_precision_score(y_true_boot, y_pred_boot))
            ef5_scores.append(calculate_enrichment_factor(y_true_boot, y_pred_boot, EF_PERCENTAGE))
            p10_scores.append(calculate_precision_at_k(y_true_boot, y_pred_boot, PRECISION_PERCENTAGE))
            tpr_at_fpr_scores.append(calculate_tpr_at_fpr(y_true_boot, y_pred_boot, TARGET_FPR))
        
    # 4. 최종 결과 테이블 생성 및 출력
    metrics = {
        "Metric": ["AUPRC", f"EF@{EF_PERCENTAGE*100:.0f}%", f"Precision@{PRECISION_PERCENTAGE*100:.0f}%", f"TPR@FPR={TARGET_FPR}"],
        "Score": [np.mean(auprc_scores), np.mean(ef5_scores), np.mean(p10_scores), np.mean(tpr_at_fpr_scores)],
        "95% CI": [np.percentile(auprc_scores, [2.5, 97.5]),
                   np.percentile(ef5_scores, [2.5, 97.5]),
                   np.percentile(p10_scores, [2.5, 97.5]),
                   np.percentile(tpr_at_fpr_scores, [2.5, 97.5])]
    }
    results_df = pd.DataFrame(metrics)
    results_df['95% CI'] = results_df['95% CI'].apply(lambda x: f"[{x[0]:.3f}, {x[1]:.3f}]")
    results_df['Score'] = results_df['Score'].apply(lambda x: f"{x:.3f}")

    print("\n\n" + "="*60)
    print("                추가 의사결정 지표 (95% CI)")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)