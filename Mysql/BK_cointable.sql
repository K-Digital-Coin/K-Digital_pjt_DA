use musthave; # 테이블 만들기 위한 DB 선택
SELECT * FROM coinproject.historycoin;
SELECT * FROM hc_bb;
# 230223 writed
# DROP TABLE hc_bb;
#SELECT * FROM coinproject.historycoin;
#update historycoin set low_price=31500000,
#                       trade_price=31583000,
#                       candle_acc_trade_volume=285.11359423
#				   where idx = 20000;

# 15hour 이평선 테이블
CREATE TABLE hc_ma (
	candle_date_time_kst DATETIME,
    ma DOUBLE,
	PRIMARY KEY(candle_date_time_kst)
)ENGINE=MYISAM CHARSET=utf8;
# RSI 테이블
CREATE TABLE hc_rsi (
	candle_date_time_kst DATETIME,
    rsi DOUBLE,
	PRIMARY KEY(candle_date_time_kst)
)ENGINE=MYISAM CHARSET=utf8;
# MACD 테이블
CREATE TABLE hc_macd (
	candle_date_time_kst DATETIME,
    macd DOUBLE,
	PRIMARY KEY(candle_date_time_kst)
)ENGINE=MYISAM CHARSET=utf8;
# 볼린저 밴드 테이블
CREATE TABLE hc_bb (
	candle_date_time_kst DATETIME,
    bbp DOUBLE,
    bbc DOUBLE,
    bbm DOUBLE,
	PRIMARY KEY(candle_date_time_kst)
)ENGINE=MYISAM CHARSET=utf8;

#참조
#CREATE TABLE bk_cointable (
#	seq	INT NOT NULL AUTO_INCREMENT,
#    candel_date_time DATETIME,
#    bk_open INT,
#    bk_high INT,
#    bk_low INT,
#    bk_close INT,
#    bk_volume FLOAT,
#    ma_c5d INT,
#    macd INT,
#    rsi FLOAT,
#    bb_p FLOAT,
#    bb_m FLOAT,
#	PRIMARY KEY(seq)
#)ENGINE=MYISAM CHARSET=utf8;#engine = mysql에서 사용하는 엔진 / charset 데이터의 캐릭터세트

#use musthave;
#drop table board;
#select * from board;
#create table member (
#	id varchar(10) not null,
#    pass varchar(10) not null,
#    name varchar(30) not null,
#    regidate timestamp default current_timestamp not null,
#    primary key (id)
#);
#select * from member;
#select * from board;
#SELECT B.*, M.name  FROM member M INNER JOIN board B  ON M.id=B.id  WHERE num=105;