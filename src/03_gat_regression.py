# 파일 경로: src/04_gat_regression_v2.py

import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 설정 부분 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
# --- 설정 끝 ---

# ★★★ 원자 특징 추출 함수 개선 ★★★
def get_atom_features(atom):
    # 원자 종류, 결합 수, 형식 전하, 수소 수, 방향족 여부 등 다양한 특징 추출
    possible_atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    atom_symbol = atom.GetSymbol()
    # 원-핫 인코딩
    atom_type_one_hot = [1 if s == atom_symbol else 0 for s in possible_atom_list]
    
    features = atom_type_one_hot + [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        atom.GetHybridization(),
        atom.GetIsAromatic(),
        atom.GetTotalNumHs(),
    ]
    return features

# 1. SMILES를 그래프 데이터 객체로 변환하는 클래스 (개선된 특징 추출기 사용)
class MoleculeDataset(Dataset):
    def __init__(self, df, target_column):
        super(MoleculeDataset, self).__init__()
        self.df = df
        self.target_column = target_column
        # 첫 번째 분자를 기준으로 특징의 개수를 결정
        self.num_node_features = self._get_num_features()

    def _get_num_features(self):
        # 첫 번째 유효한 분자의 특징 개수를 계산
        for i in range(len(self.df)):
            smiles = self.df.iloc[i]['smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return len(get_atom_features(mol.GetAtomWithIdx(0)))
        return 0 # 기본값

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        smiles = row['smiles']
        y = torch.tensor([row[self.target_column]], dtype=torch.float)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None

        atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)

        adj = GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

# 2. GAT 모델 정의 (입력 특징 개수를 동적으로 받도록 수정)
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.out = Linear(hidden_channels * num_heads, 1)

    def forward(self, data):
        # ... (이전과 동일) ...
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.2, training=self.training)
        return self.out(x)

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    print("데이터를 로드하고 그래프로 변환합니다 (v2: 풍부한 특징 사용)...")
    df = pd.read_csv(DATA_FILE_PATH)
    df.columns = [col.lower() for col in df.columns]
    target_column = 'label' if 'label' in df.columns else 'exp'
    
    dataset = MoleculeDataset(df, target_column)
    
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # DataLoader가 None을 반환하는 경우를 걸러내기 위한 collate_fn
    def collate_fn(data_list):
        batch = [data for data in data_list if data is not None]
        return torch_geometric.data.Batch.from_data_list(batch) if batch else None

    # DataLoader에 collate_fn 적용
    import torch_geometric
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # 모델 정의 시, 데이터셋에서 계산된 특징 개수를 사용
    model = GAT(num_node_features=dataset.num_node_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # 학습률 약간 감소
    criterion = torch.nn.MSELoss()
    
    # 훈련/평가 함수 (이전과 동일)
    def train():
        model.train()
        total_loss = 0
        for batch in train_loader:
            if batch is None: continue
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(train_loader.dataset)

    def test(loader):
        model.eval()
        predictions = []
        real_values = []
        with torch.no_grad():
            for batch in loader:
                if batch is None: continue
                batch = batch.to(DEVICE)
                out = model(batch)
                predictions.extend(out.cpu().numpy())
                real_values.extend(batch.y.cpu().numpy())
        return np.array(predictions), np.array(real_values)

    print("개선된 GAT 모델 훈련을 시작합니다...")
    for epoch in range(1, 151): # 에포크 수 약간 증가
        loss = train()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

    print("훈련된 모델 성능을 평가합니다...")
    y_pred, y_test = test(test_loader)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("      GAT 모델 최종 성능 (v2)")
    print("="*40)
    print(f"평균 제곱 오차 (MSE): {mse:.4f}")
    print(f"결정 계수 (R-squared): {r2:.4f}")
    print("="*40)