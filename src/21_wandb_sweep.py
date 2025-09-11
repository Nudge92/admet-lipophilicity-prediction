# 파일 경로: src/21_wandb_sweep.py (완전판)

import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import xgboost as xgb
from xgboost.callback import EarlyStopping # <--- 이 라인을 추가하세요
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import wandb # W&B 라이브러리 임포트
import yaml # YAML 파일을 읽기 위해 추가

# torch_geometric 관련 임포트
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool

# 별도 파일로 분리한 GAT 모델 및 데이터셋 클래스 임포트
from gat_regression_v4 import GAT, MoleculeDataset

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import average_precision_score

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
GAT_MODEL_PATH = "models/gat_v4_model.pt"
CHEMBERTA_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
TARGET_COLUMN = 'label'
HIT_QUANTILE = 0.8
# --- 설정 끝 ---

# --- 함수 정의 부분 (전체 포함) ---
def get_gat_embeddings(loader, model):
    model.eval()
    embeddings = []
    # tqdm 제거 (Sweep 실행 시 로그가 너무 많아짐)
    for data in loader:
        data = data.to(DEVICE)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(len(model.convs)):
            x = model.convs[i](x, edge_index)
            x = model.batch_norms[i](x)
            x = F.elu(x)
        graph_embedding = global_mean_pool(x, batch)
        embeddings.append(graph_embedding.cpu().detach().numpy())
    return np.concatenate(embeddings, axis=0)

def get_chemberta_embeddings(smiles_list, model, tokenizer):
    model.eval()
    embeddings = []
    # tqdm 제거 (Sweep 실행 시 로그가 너무 많아짐)
    for smiles in smiles_list:
        inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        outputs = model(**inputs, output_hidden_states=True)
        cls_embedding = outputs.hidden_states[-1][:, 0, :].cpu().detach().numpy()
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


# ★★★ W&B Sweep을 위한 훈련 함수 (수정 완료) ★★★
def train_for_sweep():
    # 1. W&B 초기화
    run = wandb.init()
    config = wandb.config 

    # 2. 데이터 준비 (전역 변수 사용)

    # 3. XGBoost 모델 훈련
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        random_state=42,
        n_jobs=-1
    )

    # ▼▼▼▼▼ [수정] 이 부분이 변경되었습니다 ▼▼▼▼▼
    # 구버전 방식:
    # model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

    # 신버전 방식:
    # 1. EarlyStopping 콜백 객체 생성
    # save_best=True를 추가하면 가장 성능이 좋았던 시점의 모델 가중치가 최종 모델에 저장됩니다.
    early_stopping = EarlyStopping(rounds=50, save_best=True)

    # 2. callbacks 파라미터로 전달
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=[early_stopping], # <--- 이렇게 변경
        verbose=False
    )
    # ▲▲▲▲▲ [수정] 여기까지 ▲▲▲▲▲

    # 4. 검증셋(validation set)에서 AUPRC 점수 계산
    # save_best=True를 사용했으므로, predict는 자동으로 최적의 라운드에서 수행됩니다.
    y_pred_reg = model.predict(X_val)
    val_auprc = average_precision_score(y_val_class, y_pred_reg)

    # 5. W&B에 검증 점수 기록
    wandb.log({"val_auprc": val_auprc})


# --- 메인 실행 로직 ---

# Sweep 실행 전, 데이터 로딩과 특징 추출을 딱 한 번만 수행 (전역 변수로 선언)
print("--- 데이터 로딩 및 특징 추출 (1회 수행) ---")
df = pd.read_csv(DATA_FILE_PATH)
df.columns = [col.lower() for col in df.columns]
df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
df = df.dropna(subset=['mol', TARGET_COLUMN])

gat_model = GAT(num_node_features=17).to(DEVICE)
gat_model.load_state_dict(torch.load(GAT_MODEL_PATH, map_location=DEVICE))
chemberta_model = AutoModel.from_pretrained(CHEMBERTA_CHECKPOINT).to(DEVICE)
chemberta_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_CHECKPOINT)

gat_dataset = MoleculeDataset(df, TARGET_COLUMN)
gat_loader = DataLoader(gat_dataset, batch_size=256, shuffle=False)
gat_features = get_gat_embeddings(gat_loader, gat_model)
chemberta_features = get_chemberta_embeddings(df['smiles'].tolist(), chemberta_model, chemberta_tokenizer)

X = np.concatenate([gat_features, chemberta_features], axis=1)
y = df[TARGET_COLUMN].values

# 훈련/검증/테스트 데이터 분할
train_val_idx, test_idx = next(scaffold_cv_split(df, n_splits=5, random_state=42))
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)

# train_for_sweep 함수가 접근해야 하므로 전역 변수로 유지
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

hit_threshold = np.quantile(y, HIT_QUANTILE)
y_val_class = (y_val >= hit_threshold).astype(int)

print("\n--- Sweep Agent 실행 준비 완료 ---")
# ▲▲▲▲▲ [수정] 여기까지 이동 ▲▲▲▲▲

if __name__ == '__main__':
    # 이 스크립트를 직접 실행하면 데이터 전처리 완료 후 sweep을 시작합니다.
    # train_for_sweep 함수는 wandb agent가 호출하여 사용합니다.
    
    # wandb agent 실행을 위한 sweep_id와 함수 전달
    with open('sweep_config.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    project_name = "hybrid-model-admet"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"\nSweep ID: {sweep_id}")
    print("다음 명령어를 터미널에 입력하여 Agent를 실행하세요:")
    print(f"wandb agent {wandb.api.default_entity}/{project_name}/{sweep_id}")
    
    # train_for_sweep 함수를 agent에 등록
    wandb.agent(sweep_id, function=train_for_sweep, count=10)