# 파일 경로: src/19_calibration_scaling.py

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

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
GAT_MODEL_PATH = "models/gat_v4_model.pt"
CHEMBERTA_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
TARGET_COLUMN = 'label'
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

def calculate_ece(y_true, y_pred, n_bins=15):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['bin'] = pd.cut(df['y_pred'], bins=n_bins, labels=False)
    
    bin_stats = df.groupby('bin').agg(
        count=('y_true', 'size'),
        mean_true=('y_true', 'mean'),
        mean_pred=('y_pred', 'mean')
    ).dropna()
    
    ece = np.sum(bin_stats['count'] / len(df) * np.abs(bin_stats['mean_true'] - bin_stats['mean_pred']))
    return ece

def get_calibration_curve_data(y_true, y_pred, n_bins=15):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['bin'] = pd.cut(df['y_pred'], bins=n_bins)
    
    curve_data = df.groupby('bin', observed=True).agg(
        mean_true=('y_true', 'mean'),
        mean_pred=('y_pred', 'mean')
    ).dropna()
    return curve_data['mean_pred'], curve_data['mean_true']


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

    # 2. 훈련/검증/테스트 데이터 분할 (스캐폴드 기반)
    print("\n스캐폴드 기반으로 데이터를 훈련/테스트셋으로 분할합니다...")
    train_val_idx, test_idx = next(scaffold_cv_split(df, n_splits=5, random_state=42))
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)
    
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # 3. 모델 훈련
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, 
                             learning_rate=0.05, random_state=42)
    print("\n모델을 훈련합니다...")
    model.fit(X_train, y_train)

    # 4. 최적의 온도(T) 값 찾기 (온도 스케일링)
    def temperature_scale(T, logits):
        return logits / T

    def objective(T, logits, labels):
        scaled_logits = temperature_scale(T, logits)
        return calculate_ece(labels, scaled_logits)

    print("최적의 온도를 찾습니다 (Temperature Scaling)...")
    val_logits = model.predict(X_val)
    result = minimize(objective, x0=1.0, args=(val_logits, y_val), method='L-BFGS-B', bounds=[(0.1, 5)])
    optimal_T = result.x[0]

    # 5. 보정 전/후 성능 평가
    test_logits = model.predict(X_test)
    calibrated_preds = temperature_scale(optimal_T, test_logits)

    ece_before = calculate_ece(y_test, test_logits)
    brier_before = mean_squared_error(y_test, test_logits)

    ece_after = calculate_ece(y_test, calibrated_preds)
    brier_after = mean_squared_error(y_test, calibrated_preds)

    # 6. 결과 테이블 출력
    results_data = {
        "Metric": ["ECE (↓)", "Brier Score (MSE, ↓)"],
        "Before Scaling": [f"{ece_before:.4f}", f"{brier_before:.4f}"],
        "After Scaling": [f"{ece_after:.4f}", f"{brier_after:.4f}"],
        "Improvement (%)": [f"{(ece_before - ece_after) / ece_before * 100:+.2f}%", f"{(brier_before - brier_after) / brier_before * 100:+.2f}%"]
    }
    results_df = pd.DataFrame(results_data)
    
    print("\n\n" + "="*70)
    print("                캘리브레이션 보정 전/후 성능 비교")
    print(f"(Optimal Temperature T = {optimal_T:.3f})")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)

    # 7. 캘리브레이션 곡선 시각화
    mean_pred_before, mean_true_before = get_calibration_curve_data(y_test, test_logits)
    mean_pred_after, mean_true_after = get_calibration_curve_data(y_test, calibrated_preds)
    
    plt.figure(figsize=(9, 9))
    plt.plot([min(test_logits.min(), y_test.min()), max(test_logits.max(), y_test.max())], 
             [min(test_logits.min(), y_test.min()), max(test_logits.max(), y_test.max())], 
             'r--', label='Perfectly Calibrated')
    plt.plot(mean_pred_before, mean_true_before, 'o-', label=f'Before Scaling (ECE={ece_before:.3f})')
    plt.plot(mean_pred_after, mean_true_after, 's-', label=f'After Scaling (ECE={ece_after:.3f})')
    
    plt.xlabel("Mean Predicted Value (구간별 예측값 평균)", fontsize=14)
    plt.ylabel("Mean Actual Value (구간별 실제값 평균)", fontsize=14)
    plt.title("Calibration Curve Before vs. After Scaling", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("final_calibration_scaling_plot.png")
    print("\nfinal_calibration_scaling_plot.png 파일이 저장되었습니다.")
    plt.show()