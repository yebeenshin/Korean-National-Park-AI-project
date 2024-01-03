import streamlit as st 
from streamlit_option_menu import option_menu 
import base64
import ee
import geemap

from page import home, change_detection_1, change_detection_2, change_detection_3, underwater_1, underwater_2, landslide_mod

# 이미지 경로 설정
image_path = "./streamlit/image/knps2.png"
bigleader_path = "./streamlit/image/bigleader.png"

def layout():

    st.set_page_config(page_title="반달이의 눈 : 국립공원 관리 서비스", page_icon='./streamlit/image/eye.png', layout="wide")

    # 로고 이미지 첨부(이미지 파일을 텍스트 형식으로 변환하는 함수를 정의)
    def get_image_base64(image_path):
        with open(image_path, "rb") as image_file: #'rb'는 '읽기 전용 의미
            return base64.b64encode(image_file.read()).decode()

    # 위에서 정의한 함수 세개의 이미지 파일에 적용
    encoded_image = get_image_base64(image_path)
    bigleader_encoded_image = get_image_base64(bigleader_path)

    geemap.ee_initialize()    # GEE API를 초기화합니다.

    # # HTML과 함께 이 텍스트 형식의 이미지들을 웹 페이지에 표시
    # st.markdown(f"""
    #     <div style="display: flex; margin-top: 0em; align-items: left;">
    #         <img src="data:image/png;base64,{encoded_image}" style="height: 1.8em; display: inline-block;">
    #         <img src="data:image/png;base64,{bigleader_encoded_image}" style="height: 1.6em; margin-left: 2em; display: inline-block;">
    #     </div>
    #     """, unsafe_allow_html=True)

    # 전체페이지 : Pretendard 글꼴을 위한 CSS
    font_url = "https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css"
    st.markdown(f"""
    <style>
    @import url('{font_url}');
    html, body, [class*="st-"] {{
        font-family: 'Pretendard', sans-serif;
    }}
    """, unsafe_allow_html=True)

    
    st.markdown("""
        <style>
            .block-container {
                    padding-top: 2rem;
                }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        # 사이드바에 이미지를 추가합니다.
        st.markdown(f'''
                    <div style="display: flex; margin-top: -4em; margin-bottom: 2em; align-items: center;">
                        <img src="data:image/png;base64,{encoded_image}" style="height: 1.5em; display: inline-block;">
                        <img src="data:image/png;base64,{bigleader_encoded_image}" style="height: 1.3em; margin-left: 2em; display: inline-block;">
                    ''', unsafe_allow_html=True)

        # 카테고리별 하위 메뉴 설정
        menu_items = {
            "홈": ["사용설명서"],
            "환경관리": ["지표면 변화탐지", "타임랩스"],
            "자원 모니터링": ["식생지수 분석 및 예측", "토양 프로파일링 : 성분함량과 수분특성", "수자원 관리 : 강수 및 지하수 분석"],
            "재난안전": ["산사태 예측 지도"]
        }

        if 'selected_option_menu' not in st.session_state:
            st.session_state['selected_option_menu'] = '사용설명서'

        # 사이드바에 expander 생성 및 메뉴 선택
        for category, sub_menus in menu_items.items():
            with st.sidebar.expander(category):
                for menu in sub_menus:
                    if st.button(menu):
                        st.session_state['selected_option_menu'] = menu
                        # break

    # 선택된 메뉴에 따른 함수 실행
    if st.session_state['selected_option_menu'] == "사용설명서":
        home.app()
    elif st.session_state['selected_option_menu'] == "지표면 변화탐지":
        change_detection_1.app()
    elif st.session_state['selected_option_menu'] == "식생지수 분석 및 예측":
        change_detection_2.app()
    elif st.session_state['selected_option_menu'] == "타임랩스":
        change_detection_3.app()
    elif st.session_state['selected_option_menu'] == "수자원 관리 : 강수 및 지하수 분석":
        underwater_1.app()
    elif st.session_state['selected_option_menu'] == "토양 프로파일링 : 성분함량과 수분특성":
        underwater_2.app()
    elif st.session_state['selected_option_menu'] == "산사태 예측 지도":
        landslide_mod.app()

if __name__ == "__main__":
    layout()
