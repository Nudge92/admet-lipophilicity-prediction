# 파일 경로: src/02_baseline_xgb.py (파일 이름 대소문자 문제 해결 최종본)

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import os

# --- 설정 부분 ---
# ★★★ 실제 파일 이름인 소문자 'lipophilicity.csv'로 최종 수정 ★★★
DATA_FILE_PATH = "data/raw/lipophilicity.csv" 
# --- 설정 끝 ---

# 1. 데이터 로드
print(f"데이터 파일을 로드합니다: {DATA_FILE_PATH}")
try:
    df = pd.read_csv(DATA_FILE_PATH)
except FileNotFoundError:
    print(f"오류: {DATA_FILE_PATH} 에서 파일을 찾을 수 없습니다.")
    exit()

# 2. 모든 컬럼명을 소문자로 강제 변환
df.columns = [col.lower() for col in df.columns]

# 3. 필요한 컬럼 확인
if 'smiles' not in df.columns or ('label' not in df.columns and 'exp' not in df.columns):
    print(f"오류: 파일에 'smiles'와 'label' 또는 'exp' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")
    exit()
target_column = 'label' if 'label' in df.columns else 'exp'
print(f"감지된 컬럼 -> SMILES: 'smiles', 타겟: '{target_column}'")

# 4. ECFP 핑거프린트 생성 함수
def smiles_to_ecfp(smile_string, radius=2, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smile_string)
        if mol is None: return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except: return None

# 5. 핑거프린트 변환
print("SMILES를 ECFP 핑거프린트로 변환합니다...")
df['ecfp'] = df['smiles'].apply(smiles_to_ecfp)
df = df.dropna(subset=['ecfp', target_column])
df = df.reset_index(drop=True)

# 6. 데이터 준비
X = np.array(df['ecfp'].tolist())
y = df[target_column].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. XGBoost 모델 훈련
print("XGBoost 모델을 훈련합니다...")
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
                         early_stopping_rounds=50, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# 8. 모델 평가
print("모델 성능을 평가합니다...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*40)
print("    베이스라인 모델(ECFP+XGBoost) 최종 성능")
print("="*40)
print(f"평균 제곱 오차 (MSE): {mse:.4f}")
print(f"결정 계수 (R-squared): {r2:.4f}")
print("="*40)