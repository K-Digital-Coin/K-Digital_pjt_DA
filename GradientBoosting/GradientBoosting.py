import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 데이터 로딩
data = pd.read_csv('./GradientBoosting/merged_data.csv')

# 독립 변수(feature_X)와 종속 변수(target_y) 분리
X = data.drop(['open', 'high', 'low', 'candle_date_time'], axis=1)
y = data['close']

# 학습용 데이터와 검증용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 표준화 전처리
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Gradient Boosting 모델 학습
model = GradientBoostingRegressor(
    learning_rate=0.1, n_estimators=100, subsample=1.0, 
    criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
    max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 검증용 데이터 예측
y_pred = model.predict(X_test)

# 모델의 예측 결과(y_pred)를 데이터프레임으로 변환
df = pd.DataFrame(y_pred, columns=['close'])

# 데이터프레임을 csv 파일로 저장
df.to_csv('./GradientBoosting/predictcoin.csv', index=False)

# 검증 결과 출력
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

r_squared = r2_score(y_test, y_pred)
print("R-squared: ", r_squared)