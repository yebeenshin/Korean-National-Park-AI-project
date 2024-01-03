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

    st.title("식생지수 분석 및 예측")

    st.markdown('<hr style="border:1px solid green;"/>', unsafe_allow_html=True)

    with st.expander("식생지수 분석 및 예측 사용법"):
    # 자세한 안내 메시지를 표시합니다.
        st.write("""
                1. 원하는 국립공원 선택 또는 geojson 파일 업로드로 관심 지역을 설정합니다.
                2. 'submit' 버튼을 클릭하면 식생지수 분석 시작!
                3. 10년간의 식생지수를 분석하여 미래 식생지수를 예측하고, 식생지수 추이를 보여줍니다.
                """)

    tab1, tab2 = st.tabs(['국립공원 선택', 'GeoJson 파일 업로드'])

    with tab1:
        with st.form("selected_park"):
            
            selected_park = st.selectbox("국립공원을 선택해 주세요.", national_parks['park_ko'])
            uploaded_file = None

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")

        if submitted:

            loc = change.get_aoi(selected_park, uploaded_file)
            geo_loc = loc.geometry()
            location = geo_loc.centroid().coordinates().getInfo()[::-1]

            lat, lon = location 

            with st.spinner("10년간의 식생지수 변화를 분석하고 앞으로의 변화를 예측하는 중입니다..."):   

                # Sentinel-1 이미지 컬렉션 정의
                # 위성 이미지 컬렉션을 필터링하여 가져오기
                im_coll_sub = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
                            .filterBounds(loc)
                            .filterDate(ee.Date('2014-01-01'),ee.Date('2023-12-31'))
                            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) 
                            .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                            .filter(ee.Filter.eq('instrumentMode', 'IW'))
                            .filter(ee.Filter.eq('platform_number', 'A'))
                            .map(lambda img: img.set('date', ee.Date(img.date()).format('YYYYMMdd')))
                            .sort('date'))
                
                # 식생지수 계산
                vi_collection = change.calculate_vegetation_index(im_coll_sub, loc)
                
                # 식생지수 통계 추출
                stats_collection = change.extract_statistics(vi_collection, loc)
                
                # 통계를 리스트로 변환
                stats_list = stats_collection.reduceColumns(ee.Reducer.toList(2), ['date', 'mean_rvi']).values().get(0).getInfo()
                
                # 리스트를 데이터프레임으로 변환
                df = change.stats_to_dataframe(stats_list)
                
                # Prophet 모델 생성 및 학습
                m = Prophet()
                m.fit(df)
                
                # 미래 데이터프레임 생성 및 예측
                future = m.make_future_dataframe(periods=365)
                forecast = m.predict(future)
                
                # Plotly 그래프로 시각화
                fig1 = plot_plotly(m, forecast)  # Plotly figure를 반환합니다.
                fig2 = plot_components_plotly(m, forecast)  # Plotly figure를 반환합니다.
                    
                
                st.markdown("""
                                <style>
                                    .st-emotion-cache-bro0bh {
                                            padding-top: 7em;
                                        }
                                </style>
                            """, unsafe_allow_html=True)

                with st.container():
                    col1, buff, col2 = st.columns([2, 0.2, 1])
                    with col1:
                        # Streamlit에 그래프 표시
                        st.plotly_chart(fig1, use_container_width=True)  # 인터랙티브 그래프를 표시합니다.
                    with col2:
                        st.subheader("10년간의 식생지수 변화")
                        st.markdown("""
                                    이 그래프는 위성 데이터를 통해 약 10년간의 식생지수에 대한 변화를 표현한 그래프입니다.\n
                                    검은색 점이 실제 값에 해당하며, 실제값을 기반으로 패턴을 학습하여 약 1년간의 미래 식생지수를 파란색 선으로 예측해 주고 있습니다.\n
                                    사용자가 보고 싶은 기간을 그래프의 확대/축소 기능을 통해 자세히 들여다볼 수 있습니다. 
                                    """)
                with st. container():
                    col1, buff, col2 = st.columns([2, 0.2, 1])
                    with col1:        
                        st.plotly_chart(fig2, use_container_width=True)  # 인터랙티브 그래프를 표시합니다.
                    
                    with col2:
                        st.subheader("10년간의 식생변이 추이")
                        st.markdown("""
                                    이 그래프는 위성 데이터를 통해 약 10년간의 식생지수 추이를 표현한 그래프입니다.\n
                                    2015년부터 현재까지의 추이 및 2025년까지의 예측 추이(trend)를 보여주며\n
                                    년(yearly), 주(weekly), 일(daily) 기준 추이도 보여줍니다.\n
                                    사용자가 보고 싶은 기간을 그래프의 확대/축소 기능을 통해 자세히 들여다볼 수 있습니다. 
                                    """)
        
    with tab2:
        with st.form("uploaded_file"):

            # 사용자로부터 GeoJSON 파일 업로드 받기
            uploaded_file = st.file_uploader("GeoJSON 파일을 업로드 해주세요.", type=['geojson'])
            selected_park = None

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")

        if submitted:

            loc = change.get_aoi(selected_park, uploaded_file)
            geo_loc = loc.geometry()
            location = geo_loc.centroid().coordinates().getInfo()[::-1]

            lat, lon = location 
            
            with st.spinner("10년간의 식생지수 변화를 분석하고 앞으로의 변화를 예측하는 중입니다..."):   

                # Sentinel-1 이미지 컬렉션 정의
                # 위성 이미지 컬렉션을 필터링하여 가져오기
                im_coll_sub = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
                            .filterBounds(loc)
                            .filterDate(ee.Date('2014-01-01'),ee.Date('2023-12-31'))
                            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) 
                            .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                            .filter(ee.Filter.eq('instrumentMode', 'IW'))
                            .filter(ee.Filter.eq('platform_number', 'A'))
                            .map(lambda img: img.set('date', ee.Date(img.date()).format('YYYYMMdd')))
                            .sort('date'))
                
                # 식생지수 계산
                vi_collection = change.calculate_vegetation_index(im_coll_sub, loc)
                
                # 식생지수 통계 추출
                stats_collection = change.extract_statistics(vi_collection, loc)
                
                # 통계를 리스트로 변환
                stats_list = stats_collection.reduceColumns(ee.Reducer.toList(2), ['date', 'mean_rvi']).values().get(0).getInfo()
                
                # 리스트를 데이터프레임으로 변환
                df = change.stats_to_dataframe(stats_list)
                
                # Prophet 모델 생성 및 학습
                m = Prophet()
                m.fit(df)
                
                # 미래 데이터프레임 생성 및 예측
                future = m.make_future_dataframe(periods=365)
                forecast = m.predict(future)
                
                # Plotly 그래프로 시각화
                fig1 = plot_plotly(m, forecast)  # Plotly figure를 반환합니다.
                fig2 = plot_components_plotly(m, forecast)  # Plotly figure를 반환합니다.
                    
                
                st.markdown("""
                                <style>
                                    .st-emotion-cache-bro0bh {
                                            padding-top: 6em;
                                        }
                                </style>
                            """, unsafe_allow_html=True)

                with st.container():
                    col1, buff, col2 = st.columns([2, 0.2, 1])
                    with col1:
                        # Streamlit에 그래프 표시
                        st.plotly_chart(fig1, use_container_width=True)  # 인터랙티브 그래프를 표시합니다.
                    with col2:
                        st.subheader("10년간의 식생지수 변화")
                        st.markdown("""
                                    이 그래프는 위성 데이터를 통해 약 10년간의 식생지수에 대한 변화를 표현한 그래프입니다.\n 
                                    검정색 점이 실제 값에 해당하며, 실제값을 기반으로 패턴을 학습하여 약 1년간의 미래 식생지수를 파란색 선으로 예측해주고 있습니다.\n 
                                    사용자가 보고 싶은 기간을 그래프의 확대/축소 기능을 통해 자세히 들여다 볼 수 있습니다. 
                                    """)
                with st. container():
                    col1, buff, col2 = st.columns([2, 0.2, 1])
                    with col1:        
                        st.plotly_chart(fig2, use_container_width=True)  # 인터랙티브 그래프를 표시합니다.
                    
                    with col2:
                        st.subheader("10년간의 식생변이 추세")
                        st.markdown("""
                                    이 그래프는 위성 데이터를 통해 약 10년간의 식생지수에 대한 변화를 표현한 그래프입니다.\n
                                    검정색 점이 실제 값에 해당하며, 실제값을 기반으로 패턴을 학습하여 약 1년간의 미래 식생지수를 파란색 선으로 예측해주고 있습니다.\n 
                                    사용자가 보고 싶은 기간을 그래프의 확대/축소 기능을 통해 자세히 들여다 볼 수 있습니다. 
                                    """)

if __name__ == "__main__":
    app()
