import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import joblib

# 1. 폴더 내 모든 CSV 파일 불러오기 및 병합
folder_path = 'csv_path'  # CSV 파일이 저장된 폴더 경로
file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 데이터프레임 병합
data_frames = []
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(file_path)
    data_frames.append(data)

# 모든 데이터프레임 결합
combined_data = pd.concat(data_frames, ignore_index=True)
print(f"Total combined data shape: {combined_data.shape}")

# 2. 입력 데이터(X)와 출력 데이터(y) 분리
X = combined_data[['right_elbow_angle', 'left_elbow_angle']]
y = combined_data[['right_shoulder_angle', 'left_shoulder_angle']]

# 3. 특정 조건에 맞는 데이터만 필터링
filter_mask = (X['right_elbow_angle'] > 10) & \
              (X['left_elbow_angle'] > 10) & \
              (y['right_shoulder_angle'] > 110) & \
              (y['left_shoulder_angle'] > 110)

X_filtered = X[filter_mask]
y_filtered = y[filter_mask]

print(f"Filtered data shape: {X_filtered.shape}")

# 4. 데이터셋 분할 (훈련/테스트)
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# 5. RMSE를 평가 척도로 사용하는 하이퍼파라미터 탐색
def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_score = make_scorer(rmse_scorer, greater_is_better=False)

# 하이퍼파라미터 그리드
param_grid = {
    'estimator__n_estimators': [50, 100, 150],
    'estimator__learning_rate': [0.01, 0.1, 0.2],
    'estimator__max_depth': [3, 5, 7]
}

# 기본 Gradient Boosting 모델
gbr = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))

# GridSearchCV 설정
grid_search = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    scoring=rmse_score,
    cv=3,
    verbose=3,
    n_jobs=-1
)

# 최적 하이퍼파라미터 탐색 수행
grid_search.fit(X_train, y_train)

# 결과 출력
cv_results = grid_search.cv_results_
mean_rmse = -cv_results['mean_test_score']  # Negate to get positive RMSE
std_rmse = cv_results['std_test_score']

print("\nGrid Search RMSE Results:")
for i, params in enumerate(cv_results['params']):
    print(f"Params: {params} | Mean RMSE: {mean_rmse[i]:.4f} | Std RMSE: {std_rmse[i]:.4f}")

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_rmse = np.sqrt(-grid_search.best_score_)
print(f"\nBest Parameters: {best_params}")
#print(f"Best RMSE: {best_rmse:.4f}")

# 6. 최적 모델 평가
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred, multioutput='variance_weighted')

print(f"\nTest Set Metrics:")
print(f"RMSE: Right Shoulder: {rmse[0]:.2f}, Left Shoulder: {rmse[1]:.2f}")
print(f"R2 Score (weighted): {r2:.2f}")

# 7. 성능 시각화

# 7.1. 실제 값 vs 예측 값 산점도
plt.figure(figsize=(12, 6))

# 오른쪽 어깨 vs 팔꿈치
plt.subplot(1, 2, 1)
plt.scatter(y_test['right_shoulder_angle'], X_test['right_elbow_angle'], color='blue', alpha=0.7, label='Actual Right Shoulder vs Elbow')
plt.scatter(y_pred[:, 0], X_test['right_elbow_angle'], color='red', alpha=0.7, label='Predicted Right Shoulder vs Elbow')
plt.title("Right Shoulder vs Elbow (Actual vs Predicted)", fontsize=14)
plt.xlabel("Right Shoulder Angle", fontsize=12)
plt.ylabel("Right Elbow Angle", fontsize=12)
plt.legend()
plt.grid(True)

# 왼쪽 어깨 vs 팔꿈치
plt.subplot(1, 2, 2)
plt.scatter(y_test['left_shoulder_angle'], X_test['left_elbow_angle'], color='green', alpha=0.7, label='Actual Left Shoulder vs Elbow')
plt.scatter(y_pred[:, 1], X_test['left_elbow_angle'], color='orange', alpha=0.7, label='Predicted Left Shoulder vs Elbow')
plt.title("Left Shoulder vs Elbow (Actual vs Predicted)", fontsize=14)
plt.xlabel("Left Shoulder Angle", fontsize=12)
plt.ylabel("Left Elbow Angle", fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 7.2. 잔차 플롯 (Residual Plot)
plt.figure(figsize=(12, 6))

# 오른쪽 어깨 잔차 플롯
plt.subplot(1, 2, 1)
plt.scatter(y_pred[:, 0] - y_test['right_shoulder_angle'], y_pred[:, 0], color='blue', alpha=0.7)
plt.axhline(0, color='black', linestyle='--')
plt.title("Residuals for Right Shoulder", fontsize=14)
plt.xlabel("Residuals (Predicted - Actual)", fontsize=12)
plt.ylabel("Predicted Right Shoulder Angle", fontsize=12)
plt.grid(True)

# 왼쪽 어깨 잔차 플롯
plt.subplot(1, 2, 2)
plt.scatter(y_pred[:, 1] - y_test['left_shoulder_angle'], y_pred[:, 1], color='green', alpha=0.7)
plt.axhline(0, color='black', linestyle='--')
plt.title("Residuals for Left Shoulder", fontsize=14)
plt.xlabel("Residuals (Predicted - Actual)", fontsize=12)
plt.ylabel("Predicted Left Shoulder Angle", fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()

# 8. 최적 모델 저장
model_path = "best_gbr_model.pkl"
joblib.dump(best_model, model_path)
print(f"Best model saved to {model_path}")

# 9. Grid Search 결과 시각화
plt.figure(figsize=(10, 6))
plt.errorbar(range(len(mean_rmse)), mean_rmse, yerr=std_rmse, fmt='-o', capsize=5, ecolor='red', label="Standard Deviation")
plt.title("Grid Search RMSE with Standard Deviation", fontsize=14)
plt.xlabel("Parameter Set Index", fontsize=12)
plt.ylabel("Mean RMSE (± Standard Deviation)", fontsize=12)
plt.grid(True)
plt.xticks(range(len(mean_rmse)), rotation=90, fontsize=8)
plt.tight_layout()
plt.show()


