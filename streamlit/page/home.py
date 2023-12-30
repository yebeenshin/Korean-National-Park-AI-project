import streamlit as st
import base64
import pandas as pd
from datetime import datetime, timedelta
import ee
import geemap
import os
import folium
from streamlit_folium import folium_static
import parks

def app():
    # 1/이미지를 Base64 문자열로 인코딩하는 함수
    def get_image_base64(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return "data:image/png;base64," + encoded_string

    # 2.이미지 파일을 Base64로 인코딩(그냥 경로 복붙은 작동 안됨)
    encoded_image1 = get_image_base64("./streamlit/image/homepage.png")

    st.title("반달이의 눈 : 국립공원 관리 서비스")

    st.write("")
    st.write("")

    # Markdown을 사용한 소제목
    st.markdown("##### 👍누구나 쉽게 클릭만으로, 위성 데이터 기반 국립공원 관리 서비스")
    st.write("""
            **※ 화면 창을 최대크기로 했을 때 최적의 서비스를 경험하실 수 있습니다.**
            """)


    #마크 다운바(분리 bar)
    st.markdown('<hr style="border:1px solid green;"/>', unsafe_allow_html=True)

    with st.expander("'반달이의 눈'의 의미"): # 비율로 컬럼 크기 지정
        st.write("""
                마스코트 '반달이'와 '하늘의 눈' 역할을 하는 위성 데이터를 더한 말로
                국립공원공단을 위해 성실히 자연 변화를 탐지하겠다는 저희의 포부를 담았습니다! :) 
                """)
    
    st.subheader("페이지 사용법")

    col1, buff, col2 = st.columns([2, 0.2, 1])
    with col1:
        st.markdown(f"<img src='{encoded_image1}' alt='Image1' style='width:100%'>", unsafe_allow_html=True)
        
    with col2:
        st.write("""
                    1. 초기 접속 화면 또는 홈 > 사용설명서 메뉴에서 사용법 영상 또는 해당 글을 확인합니다.\n
                    2. 페이지 좌측 사이드바에서 원하는 메뉴를 선택합니다.\n
                    3. 페이지 접속 후, 관심 지역과 탐지 기간을 설정하면 변화 탐지가 시작됩니다.
                    만약 관심 지역 또는 탐지 기간 설정창이 없는 경우에는 입력하지 않아도 변화탐지가 가능한 것입니다. 
                    """)

    st.write("")
    st.write("")
    st.write("🖥️ 각 기능별 시연영상을 [이곳](https://www.youtube.com/channel/UCtk0XQBQcN8jkxhsbL3cscQ)에서 확인하세요!")

    #마크 다운바(분리 bar)
    st.markdown('<hr style="border:1px light gray;"/>', unsafe_allow_html=True)
    st.write('※ 사용에 불편한 점이 생기면 개발자(✉️ eye.of.bandal@gmail.com )에게 편히 연락주세요.')

# launch
if __name__  == "__main__" :
    app()
