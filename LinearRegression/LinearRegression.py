from time import strptime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import torch

PATH = './LinearRegression/'
# 시세 데이터 불러오기
df = pd.read_csv(PATH + 'historydata_v1.csv')
df.set_index('candel_date_time', inplace=True) # 인덱스를 'candle_date_time' 열로 변경

# 이동평균선 계산
df['ma5'] = df['close'].rolling(window=5).mean()
df['ma10'] = df['close'].rolling(window=10).mean()
df['ma20'] = df['close'].rolling(window=20).mean()

# MACD 계산
df['ema12'] = df['close'].ewm(span=12).mean()
df['ema26'] = df['close'].ewm(span=26).mean()
df['macd'] = df['ema12'] - df['ema26']
df['signal'] = df['macd'].ewm(span=9).mean()

# RSI 계산
delta = df['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))

# 볼린저 밴드 계산
df['std20'] = df['close'].rolling(window=20).std()
df['upper_band'] = df['ma20'] + (df['std20'] * 2)
df['mid_band'] = df['ma20']
df['lower_band'] = df['ma20'] - (df['std20'] * 2)

# 모든 데이터를 포함한 csv 파일로 저장
df.to_csv(PATH + 'coin_data.csv', index=True) # 인덱스를 포함하여 저장

# # 기술지표를 csv 파일로 저장
df[['ma5']].to_csv(PATH + 'ma5.csv', index=True) # 인덱스를 포함하여 저장
df[['ma10']].to_csv(PATH + 'ma10.csv', index=True) # 인덱스를 포함하여 저장
df[['ma20']].to_csv(PATH + 'ma20.csv', index=True) # 인덱스를 포함하여 저장
df[['ema12']].to_csv(PATH + 'ema12.csv', index=True) # 인덱스를 포함하여 저장
df[['ema26']].to_csv(PATH + 'ema26.csv', index=True) # 인덱스를 포함하여 저장
df[['macd']].to_csv(PATH + 'macd.csv', index=True) # 인덱스를 포함하여 저장
df[['signal']].to_csv(PATH + 'signal.csv', index=True) # 인덱스를 포함하여 저장
df[['rsi']].to_csv(PATH + 'rsi.csv', index=True) # 인덱스를 포함하여 저장
df[['upper_band']].to_csv(PATH + 'upper_band.csv', index=True) # 인덱스를 포함하여 저장
df[['mid_band']].to_csv(PATH + 'mid_band.csv', index=True) # 인덱스를 포함하여 저장
df[['lower_band']].to_csv(PATH + 'lower_band.csv', index=True) # 인덱스를 포함하여 저장

# df = pd.read_csv(PATH + 'coin_data.csv')

# 데이터 전처리
## null 제거
dataset_raw = df[['close', 'volume', 'ma5', 'ma10', 'ma20', 'macd', 'signal', 'rsi', 'upper_band', 'mid_band', 'lower_band']]
# print(dataset_raw.isnull().sum())
dataset = dataset_raw.dropna(axis=0)
# print(dataset.isnull().sum())

## 시간 계산
last_hour = df.index[0] # 인덱스를 사용하도록 변경
datetime_format = "%Y-%m-%d %H:%M"
datetime_result = datetime.strptime(last_hour, datetime_format)

next_hour = datetime_result + timedelta(hours=1)



# ???????????????????????????????



# Min-Max 스케일링
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

n_input = 48
n_features = 11
generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)

# LSTM 모델 생성
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(generator, epochs=1)
model.summary()

# 학습한 모델 저장
# torch.save(model, PATH + 'model.pt')


# 다음 48시간 동안의 가격 예측
pred_hours = pd.date_range(start=next_hour, periods=48, freq='H')
preds = []
for i in range(48):
    x_input = scaled_data[-n_input:]
    x_input = x_input.reshape((1, n_input, n_features))
    yhat = model.predict(x_input, verbose=1)
    print(yhat)
    preds.append(yhat[0][0])
    new_data = [yhat[0][0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    scaled_data = np.concatenate((scaled_data, [new_data]))

# 결과를 csv 파일로 저장
preds_df = pd.DataFrame({'candle_date_time': pred_hours, 'predicted_price': preds})
preds_df.to_csv(PATH + 'predictcoin.csv', index=False)
