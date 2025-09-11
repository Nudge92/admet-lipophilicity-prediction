# 파일 경로: src/06_train_and_save_gat.py

import pandas as pd
import numpy as np
import os
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, BatchNorm1d
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
MODEL_SAVE_DIR = "models" # 모델을 저장할 폴더
GAT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "gat_v4_model.pt") # 최종 모델 파일 경로
# --- 설정 끝 ---

# 원자 특징 추출 함수 (제공해주신 코드와 동일)
def get_atom_features(atom):
    possible_atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'Unknown']
    atom_symbol = atom.GetSymbol()
    if atom_symbol not in possible_atom_list: atom_symbol = 'Unknown'
    atom_type_one_hot = [1 if s == atom_symbol else 0 for s in possible_atom_list]
    features = atom_type_one_hot + [
        atom.GetDegree() / 10, atom.GetFormalCharge(), float(atom.GetHybridization()),
        atom.GetIsAromatic(), atom.GetTotalNumHs() / 8,
    ]
    return features

# 데이터셋 클래스 (제공해주신 코드와 동일)
class MoleculeDataset(Dataset):
    def __init__(self, df, target_column):
        super(MoleculeDataset, self).__init__()
        self.df = df
        self.target_column = target_column
    def len(self): return len(self.df)
    def get(self, idx):
        row = self.df.iloc[idx]
        mol, y = row['mol'], torch.tensor([row[self.target_column]], dtype=torch.float)
        x = torch.tensor([get_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
        edge_indices = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, y=y)

# GAT 모델 구조 (제공해주신 코드와 동일)
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_layers=3, hidden_channels=128, num_heads=8, dropout=0.3):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        # ... (이하 동일, 생략) ...
        self.convs.append(GATv2Conv(num_node_features, hidden_channels, heads=num_heads))
        self.batch_norms.append(BatchNorm1d(hidden_channels * num_heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads))
            self.batch_norms.append(BatchNorm1d(hidden_channels * num_heads))
        self.convs.append(GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=1))
        self.batch_norms.append(BatchNorm1d(hidden_channels))
        self.out = Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.out(x)

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # 데이터 준비
    df = pd.read_csv(DATA_FILE_PATH)
    df.columns = [col.lower() for col in df.columns]
    target_column = 'label' if 'label' in df.columns else 'exp'
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol', target_column])
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset, test_dataset = MoleculeDataset(train_df, target_column), MoleculeDataset(test_df, target_column)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    num_node_features = train_dataset.num_node_features
    print(f"사용하는 기기: {DEVICE}, 원자 특징 개수: {num_node_features}")

    # 모델, 옵티마이저 등 정의 (제공해주신 코드와 동일)
    model = GAT(num_node_features=num_node_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    def train():
        model.train()
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.view_as(out))
            loss.backward()
            optimizer.step()

    def evaluate(loader):
        model.eval()
        predictions, real_values = [], []
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(DEVICE)
                out = model(data)
                predictions.extend(out.cpu().numpy().flatten())
                real_values.extend(data.y.cpu().numpy().flatten())
                loss = criterion(out, data.y.view_as(out))
                total_loss += loss.item() * data.num_graphs
        return mean_squared_error(real_values, predictions), r2_score(real_values, predictions)

    # 훈련 루프 (제공해주신 코드와 동일)
    print("GAT 모델 훈련을 시작합니다...")
    for epoch in range(1, 201):
        train()
        val_mse, val_r2 = evaluate(test_loader)
        scheduler.step(val_mse)
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch:03d}, Val R2: {val_r2:.4f}, Val MSE: {val_mse:.4f}, LR: {current_lr:.6f}")

    # 최종 성능 평가 (제공해주신 코드와 동일)
    final_mse, final_r2 = evaluate(test_loader)
    print("\n" + "="*40)
    print("      GAT 모델 최종 성능")
    print("="*40)
    print(f"평균 제곱 오차 (MSE): {final_mse:.4f}")
    print(f"결정 계수 (R-squared): {final_r2:.4f}")
    print("="*40)

    # ★★★ 모델 저장 기능 추가 ★★★
    # 'models' 폴더가 없으면 생성
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    # 훈련된 모델의 가중치(state_dict)를 저장
    torch.save(model.state_dict(), GAT_MODEL_PATH)
    print(f"\n✅ 훈련된 GAT 모델을 '{GAT_MODEL_PATH}' 경로에 성공적으로 저장했습니다.")