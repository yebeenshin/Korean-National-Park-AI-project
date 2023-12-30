import pandas as pd 

def get_parks():
    # 국립공원 리스트를 정의합니다.
    # 데이터프레임 생성
    national_parks = pd.DataFrame({
        'park_ko':['',
                    '가야산국립공원',
                    '경주국립공원',
                    '계룡산국립공원',
                    '내장산국립공원',
                    '다도해해상국립공원',
                    '덕유산국립공원',
                    '무등산국립공원',
                    '북한산국립공원',
                    '설악산국립공원',
                    '소백산국립공원',
                    '속리산국립공원',
                    '오대산국립공원',
                    '월악산국립공원',
                    '월출산국립공원',
                    '지리산국립공원',
                    '치악산국립공원',
                    '태백산국립공원',
                    '태안해안국립공원',
                    '팔공산국립공원',
                    '한려해상국립공원'],
        'park_en': ['',
                    'Gayasan',
                    'Gyeongju',
                    'Gyeryongsan',
                    'Naejangsan',
                    'Dadohaehaesang',
                    'Deogyusan',
                    'Mudeungsan',
                    'Bukhansan',
                    'Seoraksan',
                    'Sobaeksan',
                    'Songnisan',
                    'Odaesan',
                    'Woraksan',
                    'Wolchulsan',
                    'Jirisan',
                    'Chiaksan',
                    'Taebaeksan',
                    'Taeanhaean',
                    'Palgongsan',
                    'Hallyeohaesang']
    })

    return national_parks

def get_ids():
    # 지리산과 설악산의 WDPAID
    park_ids = {
        "Jirisan": 767,
        "Seoraksan": 768
    }
    return park_ids
    