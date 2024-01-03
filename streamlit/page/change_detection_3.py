import streamlit as st
import ee
import geemap
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from scipy.stats import norm, gamma, f, chi2
import IPython.display as disp
import folium
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import time
from folium.plugins import Draw
import geopandas as gpd
from streamlit_folium import folium_static, st_folium
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import os
import requests
from shapely.geometry import shape
import re
import webbrowser

import change, parks

def app():
    # folium 지도 객체에 Earth Engine 레이어 추가 메서드를 연결
    folium.Map.add_ee_layer = change.add_ee_layer

    # V-World 타일 서비스 URL (API 키 포함)
    vworld_satellite_url = "http://api.vworld.kr/req/wmts/1.0.0/{api_key}/Satellite/{z}/{y}/{x}.jpeg"
    vworld_hybrid_url = "http://api.vworld.kr/req/wmts/1.0.0/{api_key}/Hybrid/{z}/{y}/{x}.png"

    # API 키를 여기에 입력하세요
    api_key = "DCFAECAC-2343-3CB2-81AA-1FE195545C28"

    national_parks = parks.get_parks()

    max_date = datetime.today().date()
    min_date = max_date - timedelta(days=270)
                
    # gif 저장 경로 
    sar_path = 'Sentinel1.gif'
    optical_path = 'Sentinel2.gif'
    if os.path.exists(sar_path):
        os.remove(sar_path)
    
    if os.path.exists(optical_path):
        os.remove(optical_path)

    st.title("타임랩스")

    st.markdown('<hr style="border:1px solid green;"/>', unsafe_allow_html=True)
    
    with st.expander("타임랩스 사용법"):
        # 자세한 안내 메시지를 표시합니다.
        st.write('''
                1. 원하는 국립공원 선택 또는 geojson 파일 업로드로 관심 지역을 설정합니다.
                2. 관측 시작/끝 날짜와 관측 주기를 선택해 주세요. 
                3. 'submit' 버튼을 클릭하면 타임랩스 생성 시작!
                4. 선택한 기간 동안의 위성 데이터로 생성한 타임랩스를 보여줍니다.
                5. 타임랩스 하단의 다운로드 버튼을 클릭하면 타임랩스 이미지가 저장됩니다.
                ''')

    tab1, tab2 = st.tabs(['국립공원 선택', 'GeoJson 파일 업로드'])

    with tab1:
        with st.form("selected_park"):
            col1, buff, col2 = st.columns([1, 0.3, 1])

            # 현재 날짜를 한 번만 계산합니다.
            current_date = datetime.now().date()

            with col1:
                # 사용자로부터 시작 날짜를 입력 받습니다.
                start_date = st.date_input('관측 시작 날짜를 선택해 주세요.', 
                                        min_value=datetime(2014, 1, 1).date(),
                                        max_value=current_date,
                                        value=current_date - timedelta(days=150))

            with col2:
                # 사용자로부터 종료 날짜를 입력 받습니다.
                end_date = st.date_input('관측 종료 날짜를 선택해 주세요.', 
                                        value=current_date, 
                                        min_value=start_date, 
                                        max_value=current_date
                                        )
                
            if start_date >= end_date:
                st.error('종료 날짜는 시작 날짜 이후여야 합니다.')

            # GEE에서 사용할 날짜 형식으로 변환합니다.
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            
            selected_park = st.selectbox("국립공원을 선택해 주세요.", national_parks['park_ko'])
            uploaded_file = None

            freq_input = st.selectbox("관측 주기를 선택해 주세요.", ['year', 'month', 'day'])

            sat_input = st.radio(label = '사용할 이미지 데이터를 선택해 주세요.', options = ['Sentinel-1(SAR 영상)', 'Sentinel-2(광학 영상)'])
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")

        if submitted:
            
            st.markdown("""
                        사용자가 설정한 기간동안 수집된 위성 데이터를 연속적으로 보여주는 timelapse 영상입니다.\n
                        위성 영상의 해상도로 인한 화질 문제로 관심 지역 근방 10km 지역의 변화를 보여주는 영상을 제공합니다.\n
                        """)

            col1, buff, col2 = st.columns([1, 0.3, 1])

            with col1:

                # Timelapse 생성 버튼
                aoi_sub = change.get_aoi(selected_park, uploaded_file)
                with st.spinner('Timelapse를 생성하는 중입니다...'):
                        
                    if sat_input == "Sentinel-1(SAR 영상)":
                        # geemap의 sentinel1_timelapse() 메서드를 사용하여 timelapse 생성
                        timelapse = geemap.sentinel1_timelapse(
                            aoi_sub,  # 선택된 국립공원 경계
                            out_gif=sar_path,
                            start_year=start_date.year,
                            end_year=end_date.year,
                            start_date=start_date.strftime('%m-%d'),
                            end_date=end_date.strftime('%m-%d'),
                            frequency=freq_input,
                            vis_params={"min": -30, "max": 0},
                            frames_per_second=2,
                            title='Sentinel-1 Timelapse',
                            add_colorbar=True,
                            colorbar_bg_color='gray',
                            add_progress_bar=True, 
                            progress_bar_color='blue',
                        )
                    
                        if not os.path.exists(sar_path):
                            st.error('Timelapse 생성에 실패했습니다. 다시 시도해 주세요.')
                        # GIF 파일이 생성되었는지 확인
                        elif os.path.exists(sar_path):
                            # Streamlit에 GIF 표시
                            st.image(sar_path)
                            
                            # 다운로드 버튼 생성
                            with open(sar_path, 'rb') as gif_file:
                                st.download_button(
                                    label="Download Timelapse GIF",
                                    data=gif_file,
                                    file_name=sar_path,
                                    mime='image/gif'
                                )
                    else: 
                        timelapse = geemap.sentinel2_timelapse(
                                aoi_sub,
                                out_gif=optical_path,
                                start_year=start_date.year,
                                end_year=end_date.year,
                                start_date=start_date.strftime('%m-%d'),
                                end_date=end_date.strftime('%m-%d'),
                                frequency=freq_input,
                                bands=['SWIR1', 'NIR', 'Red'],
                                frames_per_second=2,
                                title='Sentinel-2 Timelapse',
                                progress_bar_color='blue',
                            )
                    
                        if not os.path.exists(optical_path):
                            st.error('Timelapse 생성에 실패했습니다. 다시 시도해 주세요.')
                        # GIF 파일이 생성되었는지 확인
                        elif os.path.exists(optical_path):
                            # Streamlit에 GIF 표시
                            st.image(optical_path)
                            
                            # 다운로드 버튼 생성
                            with open(optical_path, 'rb') as gif_file:
                                st.download_button(
                                    label="Download Timelapse GIF",
                                    data=gif_file,
                                    file_name=optical_path,
                                    mime='image/gif'
                                )
            
    with tab2:
        with st.form("uploaded_file"):
            col1, buff, col2 = st.columns([1, 0.3, 1])

            # 현재 날짜를 한 번만 계산합니다.
            current_date = datetime.now().date()

            with col1:
                # 사용자로부터 시작 날짜를 입력 받습니다.
                start_date = st.date_input('관측 시작 날짜를 선택해 주세요.', 
                                        min_value=datetime(2014, 1, 1).date(),
                                        max_value=current_date,
                                        value=current_date - timedelta(days=150))

            with col2:
                # 사용자로부터 종료 날짜를 입력 받습니다.
                end_date = st.date_input('관측 종료 날짜를 선택해 주세요.', 
                                        value=current_date, 
                                        min_value=start_date, 
                                        max_value=current_date
                                        )
                
            if start_date >= end_date:
                st.error('종료 날짜는 시작 날짜 이후여야 합니다.')

            # GEE에서 사용할 날짜 형식으로 변환합니다.
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            
            # 사용자로부터 GeoJSON 파일 업로드 받기
            uploaded_file = st.file_uploader("GeoJSON 파일을 업로드 해주세요.", type=['geojson'])
            selected_park = None

            freq_input = st.selectbox("관측 주기를 선택해 주세요.", ['year', 'month', 'day'])
            
            sat_input = st.radio(label = '사용할 이미지 데이터를 선택해 주세요..', options = ['Sentinel-1(SAR 영상)', 'Sentinel-2(광학 영상)'])
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")

        if submitted:
            
            st.markdown("""
                        사용자가 설정한 기간동안 수집된 위성 데이터를 연속적으로 보여주는 timelapse 영상입니다.\n
                        위성 영상의 해상도로 인한 화질 문제로 관심 지역 근방 10km 지역의 변화를 보여주는 영상을 제공합니다.\n
                        """)

            col1, buff, col2 = st.columns([1, 0.3, 1])

            with col1:

                # Timelapse 생성 버튼
                aoi_sub = change.get_aoi(selected_park, uploaded_file)
                with st.spinner('Timelapse를 생성하는 중입니다...'):
                        
                    if sat_input == "Sentinel-1(SAR 영상)":
                        # geemap의 sentinel1_timelapse() 메서드를 사용하여 timelapse 생성
                        timelapse = geemap.sentinel1_timelapse(
                            aoi_sub,  # 선택된 국립공원 경계
                            out_gif=sar_path,
                            start_year=start_date.year,
                            end_year=end_date.year,
                            start_date=start_date.strftime('%m-%d'),
                            end_date=end_date.strftime('%m-%d'),
                            frequency=freq_input,
                            vis_params={"min": -30, "max": 0},
                            frames_per_second=2,
                            title='Sentinel-1 Timelapse',
                            add_colorbar=True,
                            colorbar_bg_color='gray',
                            add_progress_bar=True, 
                            progress_bar_color='blue',
                        )
                    
                        if not os.path.exists(sar_path):
                            st.error('Timelapse 생성에 실패했습니다. 다시 시도해 주세요.')
                        # GIF 파일이 생성되었는지 확인
                        elif os.path.exists(sar_path):
                            # Streamlit에 GIF 표시
                            st.image(sar_path)
                            
                            # 다운로드 버튼 생성
                            with open(sar_path, 'rb') as gif_file:
                                st.download_button(
                                    label="Download Timelapse GIF",
                                    data=gif_file,
                                    file_name=sar_path,
                                    mime='image/gif'
                                )
                    else: 
                        timelapse = geemap.sentinel2_timelapse(
                                aoi_sub,
                                out_gif=optical_path,
                                start_year=start_date.year,
                                end_year=end_date.year,
                                start_date=start_date.strftime('%m-%d'),
                                end_date=end_date.strftime('%m-%d'),
                                frequency=freq_input,
                                bands=['SWIR1', 'NIR', 'Red'],
                                frames_per_second=2,
                                title='Sentinel-2 Timelapse',
                                progress_bar_color='blue',
                            )
                    
                        if not os.path.exists(optical_path):
                            st.error('Timelapse 생성에 실패했습니다. 다시 시도해 주세요.')
                        # GIF 파일이 생성되었는지 확인
                        elif os.path.exists(optical_path):
                            # Streamlit에 GIF 표시
                            st.image(optical_path)
                            
                            # 다운로드 버튼 생성
                            with open(optical_path, 'rb') as gif_file:
                                st.download_button(
                                    label="Download Timelapse GIF",
                                    data=gif_file,
                                    file_name=optical_path,
                                    mime='image/gif'
                                )

if __name__ == "__main__":
    app()
