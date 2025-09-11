# 파일 경로: src/05_gat_regression_v3.py (차원 불일치 문제 해결 최종본)

import pandas as pd
import numpy as np
import os
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, global_mean_pool # GATv2로 안정성 향상
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
# --- 설정 끝 ---

# 원자 특징 추출 함수 (이전과 동일)
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

# 데이터셋 클래스 (이전과 동일)
class MoleculeDataset(Dataset):
    def __init__(self, df, target_column):
        super(MoleculeDataset, self).__init__()
        self.df = df
        self.target_column = target_column
    def len(self):
        return len(self.df)
    def get(self, idx):
        row = self.df.iloc[idx]
        mol = row['mol']
        y = torch.tensor([row[self.target_column]], dtype=torch.float)
        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, y=y)

# GAT 모델 정의 (GATv2Conv 사용)
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, num_heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=num_heads, dropout=0.2)
        self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=0.2)
        self.out = Linear(hidden_channels * num_heads, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        return self.out(x)

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    print("데이터를 로드하고 유효한 분자만 필터링합니다...")
    df = pd.read_csv(DATA_FILE_PATH)
    df.columns = [col.lower() for col in df.columns]
    target_column = 'label' if 'label' in df.columns else 'exp'
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol', target_column])
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = MoleculeDataset(train_df, target_column)
    test_dataset = MoleculeDataset(test_df, target_column)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    num_node_features = train_dataset.num_node_features
    print(f"원자 특징의 개수: {num_node_features}")
    
    model = GAT(num_node_features=num_node_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    
    # ★★★ 훈련/평가 함수 수정 ★★★
    def train():
        model.train()
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            # ★★★ 핵심 수정: target의 차원을 out과 동일하게 맞춰줌 ★★★
            loss = criterion(out, data.y.view_as(out))
            loss.backward()
            optimizer.step()

    def test(loader):
        model.eval()
        predictions, real_values = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(DEVICE)
                out = model(data)
                predictions.extend(out.cpu().numpy())
                real_values.extend(data.y.cpu().numpy())
        return np.array(predictions), np.array(real_values)

    print("수정된 GAT 모델(v3) 훈련을 시작합니다...")
    for epoch in range(1, 201): # 에포크를 조금 더 늘려서 충분히 학습
        train()
        if epoch % 20 == 0:
            y_pred_train, y_train_real = test(train_loader)
            train_r2 = r2_score(y_train_real, y_pred_train)
            print(f"Epoch {epoch:03d}, Train R2: {train_r2:.4f}")

    print("훈련된 모델 성능을 평가합니다...")
    y_pred, y_test = test(test_loader)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("      GAT 모델 최종 성능 (v3)")
    print("="*40)
    print(f"평균 제곱 오차 (MSE): {mse:.4f}")
    print(f"결정 계수 (R-squared): {r2:.4f}")
    print("="*40)