# 파일 경로: src/10_statistical_test.py

import numpy as np
from scipy import stats

# 1. 각 모델의 5-Fold 교차 검증에서 나온 R-squared 점수들을 리스트로 만듭니다.
# (이전 실행 결과에서 복사해옴)
r2_scores_chemberta = [0.6088, 0.6428, 0.5922, 0.6033, 0.6155]
r2_scores_baseline = [0.5947, 0.5567, 0.5279, 0.5213, 0.609]

# 2. 쌍체 t-검정(Paired t-test) 수행
t_statistic, p_value = stats.ttest_rel(r2_scores_chemberta, r2_scores_baseline)

# 3. 결과 출력
print("="*40)
print("      모델 성능 통계적 유의성 검증 (t-test)")
print("="*40)
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print("="*40)

# 4. p-value 해석
if p_value < 0.05:
    print("✅ P-value가 0.05보다 작습니다.")
    print("-> 두 모델의 성능 차이는 통계적으로 유의미합니다.")
    print("-> ChemBERTa 모델의 성능 향상은 우연이 아닐 가능성이 매우 높습니다.")
else:
    print("❌ P-value가 0.05보다 큽니다.")
    print("-> 두 모델의 성능 차이가 통계적으로 유의미하다고 말하기 어렵습니다.")
    print("-> 성능 향상이 우연에 의한 결과일 수 있습니다.")