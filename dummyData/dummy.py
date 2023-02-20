import pyupbit
import pandas as pd
import pymysql

# 코인 데이터 요청해서 df에 저장
df = pyupbit.get_ohlcv("KRW-BTC", interval="minute30", to='20230215', count=50000, period=0.1)
# index로 있는 date_time을 column에 추가
df['candel_date_time'] = df.index;
# index를 번호로 지정
index_list = list(range(1, 50001))
df.index = index_list;
# df의 value 열을 삭제
df = df.drop(['value'], axis='columns')
df.to_csv('C:\\workspace\dummydata.csv', header=False, index=False)
print(df)
print(df.info())

# DB 테이블 생성
# create table dummy (
# 	idx BIGINT not null auto_increment primary key,
#     candle_date_time datetime not null,
#     open_price double not null,
#     high_price double not null,
#     low_price double not null,
#     close_price double not null,
#     volume double not null
# );

# # DB 정보
# host = "localhost"
# user = "test"
# password = 'test1234'
# database = 'coinproject'

# # DB 연결
# conn = pymysql.connect(host=host, user=user, password=password, db=database)
# curs = conn.cursor(pymysql.cursors.DictCursor)

# # DB Insert
# sql = 'INSERT INTO DUMMY (open_price, high_price, low_price, close_price, volume, candle_date_time) values(%s, %s, %s, %s, %s, %s)';

# for idx in range(len(df)):
#   curs.execute(sql, tuple(df.values[idx]))

# conn.commit()

# # 종료
# curs.close()
# conn.close()