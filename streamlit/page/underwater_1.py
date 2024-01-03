import streamlit as st
import folium
from streamlit_folium import folium_static
import geemap
import ee
import pandas as pd
import branca.colormap as cm
import matplotlib.pyplot as plt
import numpy as np
import pprint
from folium.plugins import Draw
import geopandas as gpd
from shapely.geometry import shape, Polygon
import json
import hydralit_components as hc
from datetime import datetime, timedelta
import geemap.foliumap as eefolium  # eefolium 버전 사용 

import change, parks
import matplotlib.font_manager as fm

def app():

    font_path = './font/MALGUN.TTF'
    # 폰트 프로퍼티 설정
    font_prop = fm.FontProperties(fname=font_path, size=12)
    # matplotlib의 폰트를 설정
    plt.rcParams['font.family'] = font_prop.get_name()

    # 페이지 제목 설정
    st.title("수자원 관리 : 강수 및 지하수 분석")
    st.text("""
        사용자는 선택한 지역에 대한 연평균 강수량과 지하수 재충전량을 히트맵으로 시각화한 지도, 
        월별 강수량, 잠재 증발산량, 지하수 재충전량 그래프를 통해 수자원 변화를 상세히 이해할 수 있습니다. 

        이를 통해 토양이나 식생에 물이 얼마나 공급되는지를 보여주며, 수자원 관리에 필수적인 정보를 제공합니다.
        또한, 관심 지역의 연평균 지하수 재충전량을 통해 지속 가능한 수자원 관리 및 보전 전략 수립에 있어 귀중한 기초 자료로 활용될 수 있습니다. 
    """)

    #마크 다운바(분리 bar)
    st.markdown('<hr style="border:1px solid green;"/>', unsafe_allow_html=True)

    with st.expander("수자원 관리 사용법"):
        st.text("관심 지역의 강수량 및 연간 지하수 재충전량을 확인할 수 있습니다.")
        # 자세한 안내 메시지를 표시합니다.
        st.write("""
                1. 원하는 국립공원 선택 또는 geojson 파일 업로드로 관심 지역을 설정합니다.
                2. 탐지 기간을 설정합니다.
                3. 'submit' 버튼을 클릭하면 관심 지역의 강수 및 지하수 분석을 시작합니다.
                """)

    national_parks = parks.get_parks()

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

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")

        if submitted:
            st.markdown("""
                        ✅사이드바 오른쪽 상단의 X 표시를 눌러 사이드바를 닫아주시면 최적의 서비스를 경험하실 수 있습니다.\n
                        ✅ 분석에 사용된 데이터는 1km 해상도입니다. 이는 각 1km x 1km 격자 내에서 측정된 값입니다. \n
                        　　따라서 이 값은 대규모 지역의 대표적인 수자원 수준을 나타내며,\n
                        　　특정 지점에서의 상세한 변화나 작은 규모의 데이터는 반영하지 않을 수 있습니다.
                
                        """)
            with st.spinner('강수 및 지하수 데이터를 분석하는 중입니다! 약 30초 정도 소요됩니다.'):

                
                # Add Earth Engine drawing method to folium.
                folium.Map.add_ee_layer = change.add_ee_layer

                # 피처 컬렉션을 필터링하여 한국의 특정 국립공원으로 제한
                loc = change.get_aoi(selected_park, uploaded_file)
                geo_loc = loc.geometry()
                location = geo_loc.centroid().coordinates().getInfo()[::-1]

                # 분석에 사용할 투영의 명목상의 해상도 [단위: 미터].
                scale = 1000

                # 데이터가 있는 토양 깊이 [cm].
                olm_depths = [0, 10, 30, 60, 100, 200]

                # 참조 깊이와 연결된 밴드 이름.
                olm_bands = ["b" + str(sd) for sd in olm_depths]    

                # 모래 함량과 관련된 이미지.
                sand = change.get_soil_prop("sand")

                # 점토 함량과 관련된 이미지.
                clay = change.get_soil_prop("clay")

                # 유기 탄소 함량과 관련된 이미지.
                orgc = change.get_soil_prop("orgc")

                # 유기 탄소 함량을 유기물 함량으로 변환.
                orgm = orgc.multiply(1.724)

                # 시듦점과 수분용량을 위한 두 개의 상수 이미지 초기화.
                wilting_point = ee.Image(0)
                field_capacity = ee.Image(0)

                # 각 표준 깊이에 대한 계산을 루프를 사용하여 수행.
                for key in olm_bands:
                    # 적절한 깊이에서 모래, 점토, 유기물 얻기.
                    si = sand.select(key)
                    ci = clay.select(key)
                    oi = orgm.select(key)

                    # 시듦점 계산.
                    # 주어진 깊이에 필요한 theta_1500t 매개변수.
                    theta_1500ti = (
                        ee.Image(0)
                        .expression(
                            "-0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * (S * OM)\
                            - 0.013 * (C * OM) + 0.068 * (S * C) + 0.031",
                            {
                                "S": si,
                                "C": ci,
                                "OM": oi,
                            },
                        )
                        .rename("T1500ti")
                    )

                    # 시듦점을 위한 최종 식.
                    wpi = theta_1500ti.expression(
                        "T1500ti + (0.14 * T1500ti - 0.002)", {"T1500ti": theta_1500ti}
                    ).rename("wpi")

                    # 전체 시듦점 ee.Image에 새 밴드로 추가.
                    # 데이터 타입을 float로 변환하는 것을 잊지 마세요.
                    wilting_point = wilting_point.addBands(wpi.rename(key).float())

                    # 수분용량 계산을 위한 동일한 과정.
                    # 주어진 깊이에 필요한 theta_33t 매개변수.
                    theta_33ti = (
                        ee.Image(0)
                        .expression(
                            "-0.251 * S + 0.195 * C + 0.011 * OM +\
                            0.006 * (S * OM) - 0.027 * (C * OM)+\
                            0.452 * (S * C) + 0.299",
                            {
                                "S": si,
                                "C": ci,
                                "OM": oi,
                            },
                        )
                        .rename("T33ti")
                    )

                    # 토양의 수분용량을 위한 최종 식.
                    fci = theta_33ti.expression(
                        "T33ti + (1.283 * T33ti * T33ti - 0.374 * T33ti - 0.015)",
                        {"T33ti": theta_33ti.select("T33ti")},
                    )

                    # 전체 수분용량 ee.Image에 새 밴드로 추가.
                    field_capacity = field_capacity.addBands(fci.rename(key).float())

                # 강수량 데이터 가져오기.
                pr = (
                    ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                    .select("precipitation")
                    .filterDate(start_date_str, end_date_str)
                    .filterBounds(loc)  # 선택 지역에 대한 필터링.
                )

                # 잠재 증발산량(PET)과 그 품질 지표(ET_QC) 가져오기.
                pet = (
                    ee.ImageCollection("MODIS/061/MOD16A2")
                    .select(["PET", "ET_QC"])
                    .filterDate(start_date_str, end_date_str)
                    .filterBounds(loc)  # 선택 지역에 대한 필터링.
                )

                # 강수량 데이터셋에 재샘플링 함수 적용.
                pr_m = change.sum_resampler(pr, 1, "month", 1, "pr")

                # 증발산량 데이터셋에 재샘플링 함수 적용.
                pet_m = change.sum_resampler(pet.select("PET"), 1, "month", 0.0125, "pet")

                # 강수량과 증발산량 데이터셋 병합.
                meteo = pr_m.combine(pet_m)
                zr = ee.Image(0.5)
                p = ee.Image(0.5)

                # 필드 용량과 시듦점에 함수 적용.
                fcm = change.olm_prop_mean(field_capacity, "fc_mean")
                wpm = change.olm_prop_mean(wilting_point, "wp_mean")

                # 이론적으로 사용 가능한 물 계산
                # 필드 용량 - 시듦점 차이 계산
                taw = (
                    (fcm.select("fc_mean").subtract(wpm.select("wp_mean"))).multiply(1000).multiply(zr)
                )

                # 필드 용량에 저장된 물 계산.
                stfc = taw.multiply(p)

                # 컬렉션의 시작에 따라 초기 시간(time0)을 정의합니다.
                time0 = meteo.first().get("system:time_start")

                # 토양의 수분 상태를 설명하는 모든 밴드를 초기화합니다.
                # 데이터 유형을 .float()로 형변환하는 것을 잊지 마세요.
                # 초기 재충전량 설정.
                initial_rech = ee.Image(0).set("system:time_start", time0).select([0], ["rech"]).float()

                # 누적 잠재적 수분 손실(APWL) 초기화.
                initial_apwl = ee.Image(0).set("system:time_start", time0).select([0], ["apwl"]).float()

                # 저장된 수분 초기화.
                initial_st = stfc.set("system:time_start", time0).select([0], ["st"]).float()

                # 강수량 초기화.
                initial_pr = ee.Image(0).set("system:time_start", time0).select([0], ["pr"]).float()

                # 잠재 증발산량 초기화.
                initial_pet = ee.Image(0).set("system:time_start", time0).select([0], ["pet"]).float()

                # 초기 재충전 이미지에 다른 밴드들을 추가하여 모든 밴드를 하나의 이미지로 결합합니다.
                initial_image = initial_rech.addBands(
                    ee.Image([initial_apwl, initial_st, initial_pr, initial_pet])
                )

                # 생성된 이미지를 리스트에 추가합니다. 이 리스트는 시계열 분석에서 사용됩니다.
                image_list = ee.List([initial_image])

                # 사용자 정의 함수를 기상 데이터 컬렉션에 적용.
                # rech_list = meteo.iterate(change.recharge_calculator, image_list)
                rech_list = change.call_recharge_calculator(meteo, image_list, fcm, wpm, stfc)

                # 리스트에서 초기 이미지 제거.
                rech_list = ee.List(rech_list).remove(initial_image)

                # 리스트를 ee.ImageCollection으로 변환.
                rech_coll = ee.ImageCollection(rech_list)

                # 연평균 지하수 재충전량 계산
                annual_rech = rech_coll.select("rech").mean().multiply(12)

                # 연평균 강수량 계산
                annual_pr = rech_coll.select("pr").mean().multiply(12)

                # 관심 지역 주변에 컴포지트 ee.Images 클리핑.
                rech_loc = annual_rech.clip(loc)
                pr_loc = annual_pr.clip(loc)
                
                # folium 지도 생성
                my_map = folium.Map(location=location, zoom_start=12, zoom_control=True, tiles=None)
                
                # 지하수 재충전률 시각화 파라미터 설정
                rech_vis_params = {
                    "bands": "rech",
                    "min": 0,
                    "max": 500,
                    "opacity": 1,
                    "palette": ["red", "orange", "yellow", "green", "blue", "purple"],
                }

                # 강수량 시각화 파라미터 설정
                pr_vis_params = {
                    "bands": "pr",
                    "min": 500,
                    "max": 1500,
                    "opacity": 1,
                    "palette": ["white", "blue"],
                }

                # 색상 코드에 알파 값 추가 (예: 50% 투명도)
                rech_colors_with_alpha = [color + '80' for color in rech_vis_params["palette"]]
                pr_colors_with_alpha = [color + '80' for color in pr_vis_params["palette"]]
            
                # 지하수 재충전율 컬러맵 정의
                rech_colormap = cm.LinearColormap(
                    colors=rech_vis_params["palette"],
                    caption="평균 연간 재충전율 (mm/year)",
                    vmin=rech_vis_params["min"],
                    vmax=rech_vis_params["max"],
                    
                )
                
                # 강수량 컬러맵 정의
                pr_colormap = cm.LinearColormap(
                    colors=pr_vis_params["palette"],
                    caption= "연평균 강수량 (mm/year)",
                    vmin=pr_vis_params["min"],
                    vmax=pr_vis_params["max"],  
                )
                
                # 강수량 컴포지트를 지도 객체에 추가
                my_map.add_ee_layer(pr_loc, pr_vis_params, "Precipitation")
                
                # 지하수 재충전률 컴포지트를 지도 객체에 추가
                my_map.add_ee_layer(rech_loc, rech_vis_params, "Recharge")
                
                # 컬러맵을 지도에 추가
                my_map.add_child(rech_colormap)
                my_map.add_child(pr_colormap)
    
                st.write("")
                st.write("")
                st.markdown("""
                            **:rainbow[왼쪽 범례]** : 평균 연간 재충전률(mm/year)\n
                            **:blue[오른쪽 범례]** : 연평균 강수량(mm/year)
                    
                            """)

                change.display_map_with_draw(my_map)

                ######################################################################
                # 분석에 사용할 투영의 명목상의 해상도 [단위: 미터]
                scale = 1000

                # 지하수 재충전량 컬렉션에서 관심 지역의 데이터를 배열로 가져옴.
                arr = rech_coll.getRegion(loc, scale).getInfo()

                # 배열을 pandas 데이터프레임으로 변환하고 인덱스를 정렬.
                rdf = change.ee_array_to_df(arr, ["pr", "pet", "apwl", "st", "rech"]).sort_index()
                rdf_aggregated = pd.DataFrame({
                                    'pr': rdf['pr'].resample('M').mean(),  # 월별 평균 강수량
                                    'pet': rdf['pet'].resample('M').mean(),  # 월별 평균 증발산량
                                    'rech': rdf['rech'].resample('M').sum()  # 월별 총 지하수 재충전량
                                })
                # 먼저, 이중 축을 사용하기 위한 준비를 합니다.
                fig, ax1 = plt.subplots(figsize=(15, 6))
                
                # 강수량 막대 그래프를 그립니다.
                color_pr = 'tab:blue'
                ax1.set_ylabel('강수량, 증발산량 (mm)', color='black')
                ax1.bar(rdf_aggregated.index, rdf_aggregated['pr'], color=color_pr, label='평균 강수량', alpha=0.5, width=6)
                
                # 증발산량 막대 그래프를 그립니다.
                color_pet = 'tab:orange'
                ax1.bar(rdf_aggregated.index, rdf_aggregated['pet'], color=color_pet, label='평균 증발산량', alpha=0.2, width=6)
                ax1.set_xlabel('Date')
                ax1.tick_params(axis='y', labelcolor='black')
                ax1.legend(loc='upper left')
                
                # 지하수 재충전량 그래프를 그릴 두 번째 축을 만듭니다.
                ax2 = ax1.twinx() 
                color_rech = 'tab:green' 
                ax2.set_ylabel('지하수 재충전량 (mm)', color=color_rech)
                ax2.bar(rdf_aggregated.index, rdf_aggregated['rech'], color=color_rech, label='지하수 재충전량', alpha=1, width=6)
                ax2.tick_params(axis='y', labelcolor=color_rech)
                ax2.legend(loc='upper right')
                
                # x축의 눈금을 설정합니다.
                ax1.set_xticks(rdf_aggregated.index)
                ax1.set_xticklabels(rdf_aggregated.index.strftime('%m-%Y'), rotation=90)
                
                # 타이틀을 설정합니다.
                plt.title('월별 평균 강수량, 증발산량 및 총 지하수 재충전량')
                
                # 스트림릿에 그래프 표시
                st.pyplot(fig)
                
                # pandas 데이터프레임을 연간 기준으로 재표본하여 연간 합계를 계산합니다.
                rdfy = rdf_aggregated.resample("Y").sum()
                
                # 평균값을 계산합니다.
                mean_recharge = rdfy["rech"].mean()
                
                # 결과를 출력합니다.
                st.write("우리 관심 지역의 연평균 지하수 재충전량은", int(mean_recharge), "mm/년 입니다.")

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

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")

        if submitted:
            st.markdown("""
                        ✅사이드바 오른쪽 상단의 X 표시를 눌러 사이드바를 닫아주시면 최적의 서비스를 경험하실 수 있습니다.\n
                        ✅ 분석에 사용된 데이터는 1km 해상도입니다. 이는 각 1km x 1km 격자 내에서 측정된 값입니다. \n
                        　　따라서 이 값은 대규모 지역의 대표적인 수자원 수준을 나타내며,\n
                        　　특정 지점에서의 상세한 변화나 작은 규모의 데이터는 반영하지 않을 수 있습니다.
                
                        """)
            with st.spinner('강수 및 지하수 데이터를 분석하는 중입니다! 약 30초 정도 소요됩니다.'):

                
                # Add Earth Engine drawing method to folium.
                folium.Map.add_ee_layer = change.add_ee_layer

                # 피처 컬렉션을 필터링하여 한국의 특정 국립공원으로 제한
                loc = change.get_aoi(selected_park, uploaded_file)
                geo_loc = loc.geometry()
                location = geo_loc.centroid().coordinates().getInfo()[::-1]

                # 분석에 사용할 투영의 명목상의 해상도 [단위: 미터].
                scale = 1000

                # 데이터가 있는 토양 깊이 [cm].
                olm_depths = [0, 10, 30, 60, 100, 200]

                # 참조 깊이와 연결된 밴드 이름.
                olm_bands = ["b" + str(sd) for sd in olm_depths]    

                # 모래 함량과 관련된 이미지.
                sand = change.get_soil_prop("sand")

                # 점토 함량과 관련된 이미지.
                clay = change.get_soil_prop("clay")

                # 유기 탄소 함량과 관련된 이미지.
                orgc = change.get_soil_prop("orgc")

                # 유기 탄소 함량을 유기물 함량으로 변환.
                orgm = orgc.multiply(1.724)

                # 시듦점과 수분용량을 위한 두 개의 상수 이미지 초기화.
                wilting_point = ee.Image(0)
                field_capacity = ee.Image(0)

                # 각 표준 깊이에 대한 계산을 루프를 사용하여 수행.
                for key in olm_bands:
                    # 적절한 깊이에서 모래, 점토, 유기물 얻기.
                    si = sand.select(key)
                    ci = clay.select(key)
                    oi = orgm.select(key)

                    # 시듦점 계산.
                    # 주어진 깊이에 필요한 theta_1500t 매개변수.
                    theta_1500ti = (
                        ee.Image(0)
                        .expression(
                            "-0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * (S * OM)\
                            - 0.013 * (C * OM) + 0.068 * (S * C) + 0.031",
                            {
                                "S": si,
                                "C": ci,
                                "OM": oi,
                            },
                        )
                        .rename("T1500ti")
                    )

                    # 시듦점을 위한 최종 식.
                    wpi = theta_1500ti.expression(
                        "T1500ti + (0.14 * T1500ti - 0.002)", {"T1500ti": theta_1500ti}
                    ).rename("wpi")

                    # 전체 시듦점 ee.Image에 새 밴드로 추가.
                    # 데이터 타입을 float로 변환하는 것을 잊지 마세요.
                    wilting_point = wilting_point.addBands(wpi.rename(key).float())

                    # 수분용량 계산을 위한 동일한 과정.
                    # 주어진 깊이에 필요한 theta_33t 매개변수.
                    theta_33ti = (
                        ee.Image(0)
                        .expression(
                            "-0.251 * S + 0.195 * C + 0.011 * OM +\
                            0.006 * (S * OM) - 0.027 * (C * OM)+\
                            0.452 * (S * C) + 0.299",
                            {
                                "S": si,
                                "C": ci,
                                "OM": oi,
                            },
                        )
                        .rename("T33ti")
                    )

                    # 토양의 수분용량을 위한 최종 식.
                    fci = theta_33ti.expression(
                        "T33ti + (1.283 * T33ti * T33ti - 0.374 * T33ti - 0.015)",
                        {"T33ti": theta_33ti.select("T33ti")},
                    )

                    # 전체 수분용량 ee.Image에 새 밴드로 추가.
                    field_capacity = field_capacity.addBands(fci.rename(key).float())

                # 강수량 데이터 가져오기.
                pr = (
                    ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                    .select("precipitation")
                    .filterDate(start_date_str, end_date_str)
                    .filterBounds(loc)  # 선택 지역에 대한 필터링.
                )

                # 잠재 증발산량(PET)과 그 품질 지표(ET_QC) 가져오기.
                pet = (
                    ee.ImageCollection("MODIS/061/MOD16A2")
                    .select(["PET", "ET_QC"])
                    .filterDate(start_date_str, end_date_str)
                    .filterBounds(loc)  # 선택 지역에 대한 필터링.
                )

                # 강수량 데이터셋에 재샘플링 함수 적용.
                pr_m = change.sum_resampler(pr, 1, "month", 1, "pr")

                # 증발산량 데이터셋에 재샘플링 함수 적용.
                pet_m = change.sum_resampler(pet.select("PET"), 1, "month", 0.0125, "pet")

                # 강수량과 증발산량 데이터셋 병합.
                meteo = pr_m.combine(pet_m)
                zr = ee.Image(0.5)
                p = ee.Image(0.5)

                # 필드 용량과 시듦점에 함수 적용.
                fcm = change.olm_prop_mean(field_capacity, "fc_mean")
                wpm = change.olm_prop_mean(wilting_point, "wp_mean")

                # 이론적으로 사용 가능한 물 계산
                # 필드 용량 - 시듦점 차이 계산
                taw = (
                    (fcm.select("fc_mean").subtract(wpm.select("wp_mean"))).multiply(1000).multiply(zr)
                )

                # 필드 용량에 저장된 물 계산.
                stfc = taw.multiply(p)

                # 컬렉션의 시작에 따라 초기 시간(time0)을 정의합니다.
                time0 = meteo.first().get("system:time_start")

                # 토양의 수분 상태를 설명하는 모든 밴드를 초기화합니다.
                # 데이터 유형을 .float()로 형변환하는 것을 잊지 마세요.
                # 초기 재충전량 설정.
                initial_rech = ee.Image(0).set("system:time_start", time0).select([0], ["rech"]).float()

                # 누적 잠재적 수분 손실(APWL) 초기화.
                initial_apwl = ee.Image(0).set("system:time_start", time0).select([0], ["apwl"]).float()

                # 저장된 수분 초기화.
                initial_st = stfc.set("system:time_start", time0).select([0], ["st"]).float()

                # 강수량 초기화.
                initial_pr = ee.Image(0).set("system:time_start", time0).select([0], ["pr"]).float()

                # 잠재 증발산량 초기화.
                initial_pet = ee.Image(0).set("system:time_start", time0).select([0], ["pet"]).float()

                # 초기 재충전 이미지에 다른 밴드들을 추가하여 모든 밴드를 하나의 이미지로 결합합니다.
                initial_image = initial_rech.addBands(
                    ee.Image([initial_apwl, initial_st, initial_pr, initial_pet])
                )

                # 생성된 이미지를 리스트에 추가합니다. 이 리스트는 시계열 분석에서 사용됩니다.
                image_list = ee.List([initial_image])

                # 사용자 정의 함수를 기상 데이터 컬렉션에 적용.
                # rech_list = meteo.iterate(change.recharge_calculator, image_list)
                rech_list = change.call_recharge_calculator(meteo, image_list, fcm, wpm, stfc)

                # 리스트에서 초기 이미지 제거.
                rech_list = ee.List(rech_list).remove(initial_image)

                # 리스트를 ee.ImageCollection으로 변환.
                rech_coll = ee.ImageCollection(rech_list)

                # 연평균 지하수 재충전량 계산
                annual_rech = rech_coll.select("rech").mean().multiply(12)

                # 연평균 강수량 계산
                annual_pr = rech_coll.select("pr").mean().multiply(12)

                # 관심 지역 주변에 컴포지트 ee.Images 클리핑.
                rech_loc = annual_rech.clip(loc)
                pr_loc = annual_pr.clip(loc)
                
                # folium 지도 생성
                my_map = folium.Map(location=location, zoom_start=12, zoom_control=True, tiles=None)
                
                # 지하수 재충전률 시각화 파라미터 설정
                rech_vis_params = {
                    "bands": "rech",
                    "min": 0,
                    "max": 500,
                    "opacity": 1,
                    "palette": ["red", "orange", "yellow", "green", "blue", "purple"],
                }

                # 강수량 시각화 파라미터 설정
                pr_vis_params = {
                    "bands": "pr",
                    "min": 500,
                    "max": 1500,
                    "opacity": 1,
                    "palette": ["white", "blue"],
                }

                # 색상 코드에 알파 값 추가 (예: 50% 투명도)
                rech_colors_with_alpha = [color + '80' for color in rech_vis_params["palette"]]
                pr_colors_with_alpha = [color + '80' for color in pr_vis_params["palette"]]
            
                # 지하수 재충전율 컬러맵 정의
                rech_colormap = cm.LinearColormap(
                    colors=rech_vis_params["palette"],
                    caption="평균 연간 재충전율 (mm/year)",
                    vmin=rech_vis_params["min"],
                    vmax=rech_vis_params["max"],
                    
                )
                
                # 강수량 컬러맵 정의
                pr_colormap = cm.LinearColormap(
                    colors=pr_vis_params["palette"],
                    caption= "연평균 강수량 (mm/year)",
                    vmin=pr_vis_params["min"],
                    vmax=pr_vis_params["max"],  
                )
                
                # 강수량 컴포지트를 지도 객체에 추가
                my_map.add_ee_layer(pr_loc, pr_vis_params, "Precipitation")
                
                # 지하수 재충전률 컴포지트를 지도 객체에 추가
                my_map.add_ee_layer(rech_loc, rech_vis_params, "Recharge")
                
                # 컬러맵을 지도에 추가
                my_map.add_child(rech_colormap)
                my_map.add_child(pr_colormap)
    
                st.write("")
                st.write("")
                st.markdown("""
                            **:rainbow[왼쪽 범례]** : 평균 연간 재충전률(mm/year)\n
                            **:blue[오른쪽 범례]** : 연평균 강수량(mm/year)
                    
                            """)

                change.display_map_with_draw(my_map)

                ######################################################################
                # 분석에 사용할 투영의 명목상의 해상도 [단위: 미터]
                scale = 1000

                # 지하수 재충전량 컬렉션에서 관심 지역의 데이터를 배열로 가져옴.
                arr = rech_coll.getRegion(loc, scale).getInfo()

                # 배열을 pandas 데이터프레임으로 변환하고 인덱스를 정렬.
                rdf = change.ee_array_to_df(arr, ["pr", "pet", "apwl", "st", "rech"]).sort_index()
                rdf_aggregated = pd.DataFrame({
                                    'pr': rdf['pr'].resample('M').mean(),  # 월별 평균 강수량
                                    'pet': rdf['pet'].resample('M').mean(),  # 월별 평균 증발산량
                                    'rech': rdf['rech'].resample('M').sum()  # 월별 총 지하수 재충전량
                                })
                # 먼저, 이중 축을 사용하기 위한 준비를 합니다.
                fig, ax1 = plt.subplots(figsize=(15, 6))
                
                # 강수량 막대 그래프를 그립니다.
                color_pr = 'tab:blue'
                ax1.set_ylabel('강수량, 증발산량 (mm)', color='black')
                ax1.bar(rdf_aggregated.index, rdf_aggregated['pr'], color=color_pr, label='평균 강수량', alpha=0.5, width=6)
                
                # 증발산량 막대 그래프를 그립니다.
                color_pet = 'tab:orange'
                ax1.bar(rdf_aggregated.index, rdf_aggregated['pet'], color=color_pet, label='평균 증발산량', alpha=0.2, width=6)
                ax1.set_xlabel('Date')
                ax1.tick_params(axis='y', labelcolor='black')
                ax1.legend(loc='upper left')
                
                # 지하수 재충전량 그래프를 그릴 두 번째 축을 만듭니다.
                ax2 = ax1.twinx() 
                color_rech = 'tab:green' 
                ax2.set_ylabel('지하수 재충전량 (mm)', color=color_rech)
                ax2.bar(rdf_aggregated.index, rdf_aggregated['rech'], color=color_rech, label='지하수 재충전량', alpha=1, width=6)
                ax2.tick_params(axis='y', labelcolor=color_rech)
                ax2.legend(loc='upper right')
                
                # x축의 눈금을 설정합니다.
                ax1.set_xticks(rdf_aggregated.index)
                ax1.set_xticklabels(rdf_aggregated.index.strftime('%m-%Y'), rotation=90)
                
                # 타이틀을 설정합니다.
                plt.title('월별 평균 강수량, 증발산량 및 총 지하수 재충전량')
                
                # 스트림릿에 그래프 표시
                st.pyplot(fig)
                
                # pandas 데이터프레임을 연간 기준으로 재표본하여 연간 합계를 계산합니다.
                rdfy = rdf_aggregated.resample("Y").sum()
                
                # 평균값을 계산합니다.
                mean_recharge = rdfy["rech"].mean()
                
                # 결과를 출력합니다.
                st.write("우리 관심 지역의 연평균 지하수 재충전량은", int(mean_recharge), "mm/년 입니다.")



if __name__ == "__main__":
    app()
