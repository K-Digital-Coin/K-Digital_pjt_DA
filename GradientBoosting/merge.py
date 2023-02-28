import pandas as pd
import glob

# 시세 데이터 불러오기
df_history = pd.read_csv('./dummyData/historydata_v1.csv')

# 읽어올 CSV 파일들이 있는 폴더 경로 설정
path = './data/'

# 폴더 내의 모든 CSV 파일 경로 리스트 생성
indicators = glob.glob(f'{path}/*.csv')

# CSV 파일들을 DataFrame 객체로 읽어오기
data_frames = [df_history] + [pd.read_csv(indicator) for indicator in indicators]
print(data_frames)
# 특정 열을 기준으로 병합하기 위해 공통 열 이름 설정
join_key = 'candel_date_time'

# 공통 열을 기준으로 DataFrame 객체들을 병합
merged_data = pd.merge(data_frames[0], data_frames[1], on=join_key)
for i in range(2, len(data_frames)):
    merged_data = pd.merge(merged_data, data_frames[i], on=join_key)

# 결과 파일로 저장
merged_data.to_csv('./GradientBoosting/merged_data.csv', index=False)

print('데이터 병합이 완료되었습니다.')
