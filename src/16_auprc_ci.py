# 파일 경로: src/16_auprc_ci.py

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
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
GAT_MODEL_PATH = "models/gat_v4_model.pt"
CHEMBERTA_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
TARGET_COLUMN = 'label'
HIT_QUANTILE = 0.8 # 상위 20%를 'Hit'으로 정의
N_BOOTSTRAPS = 1000 # 신뢰구간 계산을 위한 부트스트랩 반복 횟수
# --- 설정 끝 ---

# --- 함수 정의 부분 (이전과 동일, 전체 포함) ---
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

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # 1. 데이터 및 특징 준비 (이전과 동일)
    # ... (생략된 부분은 15_decision_metrics.py 와 동일) ...
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

    # 4. 분류 문제로 변환
    hit_threshold = np.quantile(y, HIT_QUANTILE)
    y_true_class = (y_test >= hit_threshold).astype(int)

    # ★★★ 5. 부트스트랩을 이용한 AUPRC 및 신뢰구간 계산 ★★★
    print(f"\n부트스트랩을 시작합니다 (반복 횟수: {N_BOOTSTRAPS})...")
    auprc_scores = []
    n_test = len(y_test)
    test_indices = np.arange(n_test)
    
    for i in tqdm(range(N_BOOTSTRAPS)):
        # 테스트셋에서 복원 추출하여 부트스트랩 샘플 생성
        bootstrap_indices = np.random.choice(test_indices, size=n_test, replace=True)
        y_true_boot = y_true_class[bootstrap_indices]
        y_pred_boot = y_pred_reg[bootstrap_indices]
        
        # AUPRC 계산 및 저장
        auprc = average_precision_score(y_true_boot, y_pred_boot)
        auprc_scores.append(auprc)
        
    mean_auprc = np.mean(auprc_scores)
    # 95% 신뢰구간 계산 (2.5번째 백분위수, 97.5번째 백분위수)
    confidence_interval = np.percentile(auprc_scores, [2.5, 97.5])
    
    # 6. 결과 출력
    print("\n" + "="*50)
    print("        AUPRC (PR-AUC) 및 95% 신뢰구간")
    print("="*50)
    print(f"평균 AUPRC: {mean_auprc:.4f}")
    print(f"95% 신뢰구간 (CI): [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
    print("="*50)

    # 7. PR 커브 시각화
    precision, recall, _ = precision_recall_curve(y_true_class, y_pred_reg)
    plt.figure(figsize=(9, 9))
    plt.plot(recall, precision, lw=2, label=f'Final Hybrid Model (AUPRC = {mean_auprc:.2f})')
    plt.xlabel('Recall (재현율)', fontsize=14)
    plt.ylabel('Precision (정밀도)', fontsize=14)
    plt.title('Precision-Recall Curve for Final Hybrid Model', fontsize=16)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig("final_pr_curve.png")
    print("\nfinal_pr_curve.png 파일이 저장되었습니다.")
    plt.show()