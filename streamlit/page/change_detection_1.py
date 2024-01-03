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

    # # V-World 타일 서비스 URL (API 키 포함)
    # vworld_satellite_url = "http://api.vworld.kr/req/wmts/1.0.0/{api_key}/Satellite/{z}/{y}/{x}.jpeg"
    # vworld_hybrid_url = "http://api.vworld.kr/req/wmts/1.0.0/{api_key}/Hybrid/{z}/{y}/{x}.png"

    # # API 키를 여기에 입력하세요
    # api_key = "DCFAECAC-2343-3CB2-81AA-1FE195545C28"

    national_parks = parks.get_parks()

    max_date = datetime.today().date()
    min_date = max_date - timedelta(days=270)

    st.title("지표면 변화탐지")
    st.text("관심 지역의 지표면 변화를 확인할 수 있습니다.")
    
    st.markdown('<hr style="border:1px solid green;"/>', unsafe_allow_html=True)
    
    with st.expander("지표면 변화탐지 사용법"):
    # 자세한 안내 메시지를 표시합니다.
        st.write('''
                1. 원하는 국립공원 선택 또는 geojson 파일 업로드로 관심 지역을 설정합니다.
                2. 탐지 기간을 설정합니다.
                3. 'submit' 버튼을 클릭하면 지표면 변화탐지 시작!
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
                                        value=current_date - timedelta(days=150))

            # 종료 날짜의 최대 값을 계산합니다.
            max_end_date = min(start_date + timedelta(days=270), current_date)

            with col2:
                # 사용자로부터 종료 날짜를 입력 받습니다.
                end_date = st.date_input('관측 종료 날짜를 선택해 주세요.', 
                                        value=current_date, 
                                        min_value=start_date, 
                                        max_value=max_end_date
                                        )

            # 날짜 간격이 9개월을 초과하는지 확인합니다.
            if start_date < end_date and (end_date - start_date).days > 270:
                st.error('선택한 기간이 9개월을 초과합니다. 날짜를 다시 선택해 주세요.')
                
            if start_date >= end_date:
                st.error('종료 날짜는 시작 날짜 이후여야 합니다.')

            # GEE에서 사용할 날짜 형식으로 변환합니다.
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            
            selected_park = st.selectbox("국립공원을 선택해 주세요.", national_parks['park_ko'])
            uploaded_file = None

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")

        if submitted:

            my_bar = st.progress(0)

            loc = change.get_aoi(selected_park, uploaded_file)
            geo_loc = loc.geometry()
            location = geo_loc.centroid().coordinates().getInfo()[::-1]

            with st.spinner('변화 탐지가 진행 중입니다. 잠시만 기다려 주세요...'):
                
                # 위성 이미지 컬렉션을 필터링하여 가져오기
                im_coll = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
                        .filterBounds(loc)
                        .filterDate(ee.Date(start_date_str),ee.Date(end_date_str))
                        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) 
                        .filter(ee.Filter.eq('instrumentMode', 'IW'))
                        .filter(ee.Filter.eq('platform_number', 'A'))
                        .map(lambda img: img.set('date', ee.Date(img.date()).format('YYYYMMdd')))
                        .sort('date'))

                # 타임스태프 목록 생성
                timestamplist = (im_coll.aggregate_array('date')
                                .map(lambda d: ee.String('T').cat(ee.String(d)))
                                .getInfo())

                # 이미지 리스트를 관심지역으로 자르는 함수
                def clip_img(img):
                    return ee.Image(img).clip(loc)
                
                # 이미지 컬렉션을 리스트로 변환하고 각 이미지를 관심 지역에 맞게 자르기
                im_list = im_coll.toList(im_coll.size())
                im_list = ee.List(im_list.map(clip_img))
                # 변화 방향을 포함한 테마별 변화지도 계산
                result = ee.Dictionary(change.change_maps(im_list, median=True, alpha=0.01))

                mp = folium.Map(location=location, zoom_start=15, tiles=None)
                
                # 관심 지역의 경계를 지도에 추가합니다.
                aoi_geojson = geemap.ee_to_geojson(loc)
                folium.GeoJson(
                    data=aoi_geojson,
                    name='AOI Boundary',
                    style_function=lambda x: {'color': 'red', 'weight': 1, 'fillOpacity': 0}
                ).add_to(mp)
                
                # Extract the change maps and export to assets.
                cmap = ee.Image(result.get('cmap'))
                smap = ee.Image(result.get('smap'))
                fmap = ee.Image(result.get('fmap'))
                bmap = ee.Image(result.get('bmap'))
                cmaps = ee.Image.cat(cmap, smap, fmap, bmap).rename(['cmap', 'smap', 'fmap'] + timestamplist[1:])
                
                # 변화가 있는 영역만 마스킹
                cmaps = cmaps.updateMask(cmaps.gt(0))
                palette = ['black', 'red', 'cyan']
                # cmaps 이미지에서 모든 밴드 이름을 가져옴
                band_names = cmaps.bandNames().getInfo()
                # folium 지도에 레이어 추가 및 진행 상태 업데이트
                total_layers = len(band_names)
                for index, band_name in enumerate(band_names):
                    if band_name.startswith('T'):
                        mp.add_ee_layer(cmaps.select(band_name), {'min': 0, 'max': 3, 'palette': palette}, band_name)
            
                        # 진행 상태를 업데이트합니다.
                        progress = int((index + 1) / total_layers * 100)
                        my_bar.progress(progress)
                
            
            st.divider() 
                
            st.success('변화 탐지가 완료되었습니다!')  
            col1, buff, col2 = st.columns([2.5, 0.1, 1])      
            
            with col1:
                # 저장된 지도를 Streamlit 앱에 표시합니다.
                change.display_map_with_draw(mp)
            with col2:
                st.subheader("🛰 Sentinel-1 위성 변화 탐지 지도 데이터 해석")
                st.markdown("""
                        지표면 반사율 변화를 통한 지형 및 생태계 변화 탐지
                        """)
                st.divider()
                st.markdown("""
                        **색상 코드**\n 
                        반사율 증가(:red[빨간색]), 반사율 감소(:blue[파란색]))
                        """)

                st.divider()

                st.markdown("""
                            
                        **반사율이 증가하는 경우**
                        1. 식생이 더 무성해질 때 
                        2. 토양이나 식물에 수분이 더 많이 함유될 때(습할 때) 
                        3. 인공 구조물이 생겨나거나 변화할 때 등\n
                        가장 쉽게 추측할 수 있는 변화로는 해당 지역 강수로 인해 물웅덩이가 생겨나면서 반사율이 증가했다는 것입니다.
                        """)

                st.divider()

                st.markdown("""
                        **반사율이 감소하는 경우** 
                        1. 식생이 감소했을 때 
                        2. 토양이 건조해질 때 등\n
                        특정 지역에서 연속적으로 변화가 감지된다면, 이는 해당 지역의 잦은 지형 변화를 의미하며 눈여겨볼 필요가 있는 집중관리 지역에 해당합니다.
                        """)
    
    with tab2:
        with st.form("uploaded_file"):
            col1, buff, col2 = st.columns([1, 0.3, 1])
            
            # 현재 날짜를 한 번만 계산합니다.
            current_date = datetime.now().date()

            with col1:
                # 사용자로부터 시작 날짜를 입력 받습니다.
                start_date = st.date_input('관측 시작 날짜를 선택해 주세요.', 
                                        min_value=datetime(2014, 1, 1).date(), 
                                        value=current_date - timedelta(days=150))

            # 종료 날짜의 최대 값을 계산합니다.
            max_end_date = min(start_date + timedelta(days=270), current_date)

            with col2:
                # 사용자로부터 종료 날짜를 입력 받습니다.
                end_date = st.date_input('관측 종료 날짜를 선택해 주세요.', 
                                        value=current_date, 
                                        min_value=start_date, 
                                        max_value=max_end_date
                                        )

            # 날짜 간격이 9개월을 초과하는지 확인합니다.
            if start_date < end_date and (end_date - start_date).days > 270:
                st.error('선택한 기간이 9개월을 초과합니다. 날짜를 다시 선택해 주세요.')
                
            if start_date >= end_date:
                st.error('종료 날짜는 시작 날짜 이후여야 합니다.')

            # GEE에서 사용할 날짜 형식으로 변환합니다.
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            
            # 사용자로부터 GeoJSON 파일 업로드 받기
            uploaded_file = st.file_uploader("GeoJSON 파일을 업로드 해주세요.", type=['geojson'])
            selected_park = None

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")

        if submitted:

            my_bar = st.progress(0)

            loc = change.get_aoi(selected_park, uploaded_file)
            geo_loc = loc.geometry()
            location = geo_loc.centroid().coordinates().getInfo()[::-1]

            with st.spinner('변화 탐지가 진행 중입니다. 잠시만 기다려 주세요...'):
                
                # 위성 이미지 컬렉션을 필터링하여 가져오기
                im_coll = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
                        .filterBounds(loc)
                        .filterDate(ee.Date(start_date_str),ee.Date(end_date_str))
                        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) 
                        .filter(ee.Filter.eq('instrumentMode', 'IW'))
                        .filter(ee.Filter.eq('platform_number', 'A'))
                        .map(lambda img: img.set('date', ee.Date(img.date()).format('YYYYMMdd')))
                        .sort('date'))

                # 타임스태프 목록 생성
                timestamplist = (im_coll.aggregate_array('date')
                                .map(lambda d: ee.String('T').cat(ee.String(d)))
                                .getInfo())

                # 이미지 리스트를 관심지역으로 자르는 함수
                def clip_img(img):
                    return ee.Image(img).clip(loc)
                
                # 이미지 컬렉션을 리스트로 변환하고 각 이미지를 관심 지역에 맞게 자르기
                im_list = im_coll.toList(im_coll.size())
                im_list = ee.List(im_list.map(clip_img))
                # 변화 방향을 포함한 테마별 변화지도 계산
                result = ee.Dictionary(change.change_maps(im_list, median=True, alpha=0.01))

                mp = folium.Map(location=location, zoom_start=15, tiles=None)
                
                # 관심 지역의 경계를 지도에 추가합니다.
                aoi_geojson = geemap.ee_to_geojson(loc)
                folium.GeoJson(
                    data=aoi_geojson,
                    name='AOI Boundary',
                    style_function=lambda x: {'color': 'red', 'weight': 1, 'fillOpacity': 0}
                ).add_to(mp)
                
                # Extract the change maps and export to assets.
                cmap = ee.Image(result.get('cmap'))
                smap = ee.Image(result.get('smap'))
                fmap = ee.Image(result.get('fmap'))
                bmap = ee.Image(result.get('bmap'))
                cmaps = ee.Image.cat(cmap, smap, fmap, bmap).rename(['cmap', 'smap', 'fmap'] + timestamplist[1:])
                
                # 변화가 있는 영역만 마스킹
                cmaps = cmaps.updateMask(cmaps.gt(0))
                palette = ['black', 'red', 'cyan']
                # cmaps 이미지에서 모든 밴드 이름을 가져옴
                band_names = cmaps.bandNames().getInfo()
                # folium 지도에 레이어 추가 및 진행 상태 업데이트
                total_layers = len(band_names)
                for index, band_name in enumerate(band_names):
                    if band_name.startswith('T'):
                        mp.add_ee_layer(cmaps.select(band_name), {'min': 0, 'max': 3, 'palette': palette}, band_name)
            
                        # 진행 상태를 업데이트합니다.
                        progress = int((index + 1) / total_layers * 100)
                        my_bar.progress(progress)
                
            
            st.divider() 
                
            st.success('변화 탐지가 완료되었습니다!')  
            col1, buff, col2 = st.columns([2, 0.1, 1])      
            
            with col1:
                # 저장된 지도를 Streamlit 앱에 표시합니다.
                change.display_map_with_draw(mp)
            with col2:
                st.subheader("🛰 Sentinel-1 위성 변화 탐지 지도 데이터 해석")
                st.markdown("""
                        지표면 반사율 변화를 통한 지형 및 생태계 변화 탐지
                        """)
                st.divider()
                st.markdown("""
                        **색상 코드**\n 
                        반사율 증가(:red[빨간색]), 반사율 감소(:blue[파란색]))
                        """)

                st.divider()

                st.markdown("""
                            
                        **반사율이 증가하는 경우**
                        1. 식생이 더 무성해질 때 
                        2. 토양이나 식물에 수분이 더 많이 함유될 때(습할 때) 
                        3. 인공 구조물이 생겨나거나 변화할 때 등\n
                        가장 쉽게 추측할 수 있는 변화로는 해당 지역 강수로 인해 물웅덩이가 생겨나면서 반사율이 증가했다는 것입니다.
                        """)

                st.divider()

                st.markdown("""
                        **반사율이 감소하는 경우** 
                        1. 식생이 감소했을 때 
                        2. 토양이 건조해질 때 등\n
                        특정 지역에서 연속적으로 변화가 감지된다면, 이는 해당 지역의 잦은 지형 변화를 의미하며 눈여겨볼 필요가 있는 집중관리 지역에 해당합니다.
                        """)

if __name__ == "__main__":
    app()
