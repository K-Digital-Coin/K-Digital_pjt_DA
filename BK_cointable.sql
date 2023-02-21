use coinproject; # 테이블 만들기 위한 DB 선택
CREATE TABLE bk_cointable (
	seq	INT NOT NULL AUTO_INCREMENT,
    candel_date_time DATETIME,
    bk_open INT,
    bk_high INT,
    bk_low INT,
    bk_close INT,
    bk_volume FLOAT,
    ma_c5d INT,
    macd INT,
    rsi FLOAT,
    bb_p FLOAT,
    bb_m FLOAT,
    PRIMARY KEY(seq)
)ENGINE=MYISAM CHARSET=utf8;#engine = mysql에서 사용하는 엔진 / charset 데이터의 캐릭터세트