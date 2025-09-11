# 파일 경로: src/09_baseline_model_cv.py (베이스라인 모델)

import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import KFold, cross_val_score

# --- 설정 부분 ---
DATA_FILE_PATH = "data/raw/lipophilicity.csv"
TARGET_COLUMN = 'label'
# --- 설정 끝 ---


def generate_ecfp(mol, radius=2, n_bits=2048):
    """RDKit 분자 객체로부터 ECFP(Morgan Fingerprint)를 생성합니다."""
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


# --- 메인 실행 로직 (교차 검증) ---
if __name__ == '__main__':
    # 1. 전체 데이터 준비
    print("데이터를 불러옵니다...")
    df = pd.read_csv(DATA_FILE_PATH)
    df.columns = [col.lower() for col in df.columns]

    # RDKit 분자 객체 생성 및 유효하지 않은 데이터 제거
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol', TARGET_COLUMN])
    print(f"전체 데이터셋 크기: {len(df)}")

    # 2. ECFP 특징 추출
    print("\nECFP 특징을 추출합니다...")
    # tqdm을 사용하여 진행 상황 표시
    tqdm.pandas(desc="ECFP 생성 중")
    X_ecfp = df['mol'].progress_apply(generate_ecfp)
    
    # 리스트로 된 특징들을 하나의 numpy 배열로 변환
    X = np.stack(X_ecfp.values)
    y = df[TARGET_COLUMN].values

    # 3. 5-Fold 교차 검증을 위한 최종 모델 및 CV 객체 정의
    final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, 
                                   learning_rate=0.05, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 4. 교차 검증 실행
    print("\n--- 5-Fold 교차 검증 시작 ---")
    r2_scores = cross_val_score(final_model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    mse_scores = cross_val_score(final_model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1) * -1
    
    # 5. 최종 결과 출력
    print("\n" + "="*40)
    print("      베이스라인 모델(ECFP+XGBoost) 교차 검증 최종 성능")
    print("="*40)
    print(f"각 Fold의 R-squared: {np.round(r2_scores, 4)}")
    print(f"-> 평균 R-squared: {np.mean(r2_scores):.4f} (표준편차: {np.std(r2_scores):.4f})\n")
    print(f"각 Fold의 MSE: {np.round(mse_scores, 4)}")
    print(f"-> 평균 MSE: {np.mean(mse_scores):.4f} (표준편차: {np.std(mse_scores):.4f})")
    print("="*40)