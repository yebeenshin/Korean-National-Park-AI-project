import streamlit as st
import ee
import pandas as pd 
import folium
from folium.plugins import Draw
from streamlit_folium import folium_static
import streamlit.components.v1 as components
import change, parks
import geemap.foliumap as geemap

def app():
    # V-World 타일 서비스 URL (API 키 포함)
    vworld_satellite_url = "http://api.vworld.kr/req/wmts/1.0.0/{api_key}/Satellite/{z}/{y}/{x}.jpeg"
    vworld_hybrid_url = "http://api.vworld.kr/req/wmts/1.0.0/{api_key}/Hybrid/{z}/{y}/{x}.png"
    
    # API 키를 여기에 입력하세요
    api_key = "DCFAECAC-2343-3CB2-81AA-1FE195545C28"

    # 페이지 제목 설정
    st.title("산사태 예측 지도")

    #마크 다운바(분리 bar)
    st.markdown('<hr style="border:1px solid green;"/>', unsafe_allow_html=True)# 스트림릿 앱에서 표시
    
    with st.expander("산사태 예측 지도 사용법"):
        st.markdown(f"""
        <iframe width=100% height="600"str src="https://www.youtube.com/embed/spHn9tfebk0?si=JtPMVGwYroJ7-x7k" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        """, unsafe_allow_html=True)

        # 자세한 안내 메시지를 표시합니다.
        st.write("""
                1. 산사태 예측 지도 페이지 접속 후, 보고싶은 권역을 선택합니다
                2. 예측 지도를 클릭하면 해당 지점의 위도/경도를 확인할 수 있습니다.
                3. 지도 하단의 입력창에 위도와 경도를 입력하면, 어떤 요소가 산사태에 영향을 미칠지 확인할 수 있습니다.
                """)

    # if 'region' not in st.session_state:
    #     st.session_state['region'] = ''

    # with st.form("selected_region"):
    #     # 권역 선택 드롭다운 메뉴
        # st.session_state['region'] = st.selectbox(
        #     '보고싶은 권역을 선택해주세요:',
        #     ('', '수도권', '영남권', '호남권', '충청권', '강원권')
        # )

    region = st.selectbox(
        '보고싶은 권역을 선택해주세요:',
        ('', '수도권', '영남권', '호남권', '충청권', '강원권')
    )

    if region != '':

        model_average_file, data_file, threshold, variables, importance, auc_roc, auc_pr = change.get_region_data(region)

        st.write('')
        col1, buff, col2 = st.columns([1, 0.2, 1])
        
        with col1:
            # 예측 정확도 표시###############################################################################
            st.subheader('산사태 예측 정확도')
            st.write(f"AUC-ROC: {auc_roc}")
            st.write(f"AUC-PR: {auc_pr}")
        with col2:
            # 지도 설명 
            st.subheader('산사태 예측 지도 설명')
            st.markdown('''
                        주황색점: 과거 산사태 발생 지점\n
                        위험도 범주: 빨간색에 가까울수록 산사태 발생 확률이 높은 지점
                        ''')
        
        st.write('')
        
        with st.spinner("산사태 예측지도 생성 중입니다! 약 5분 정도 소요됩니다."):                
            national_parks = parks.get_parks()
            
            # 파일 로드
            ModelAverage = pd.read_pickle(model_average_file)
            Data = pd.read_pickle(data_file)
            
            # folium 지도 객체에 Earth Engine 레이어 추가 메서드를 연결
            folium.Map.add_ee_layer = change.add_ee_layer

            # 임계값 이상의 값을 가지는 영역에 대한 그라데이션 적용
            # 임계값 미만의 값은 마스크 처리되어 표시되지 않음
            masked_image = ModelAverage.updateMask(ModelAverage.gt(threshold))

            # 최적 임계값을 기반으로 사용자 정의 이진 분포 지도화
            vis_params = {
                'min': threshold,  # 최소값을 임계값으로 설정
                'max': 1,          # 최대값
                'palette': ['#00FF00', '#FFFF00', '#FF0000'],
                'opacity' : 0.6,
                'transparent': True,}

            Map = geemap.Map()

            # # V-World 타일 레이어 추가
            folium.TileLayer(tiles=vworld_satellite_url.replace("{api_key}", api_key),
                            attr='V-World Satellite',
                            name='V-World Satellite', overlay = False).add_to(Map)
                            
            folium.TileLayer(tiles=vworld_hybrid_url.replace("{api_key}", api_key),
                            attr='V-World Hybrid',
                            name='V-World Hybrid', overlay = True).add_to(Map)
            
            Map.addLayer(masked_image, vis_params, 'Landslide vulnerability (Thresholded)')
            
            # 컬러바 추가
            Map.add_colorbar(vis_params, label="Landslide vulnerability", orientation="horizontal", layer_name="Landslide vulnerability (Thresholded)")
            
            # 산사태 발생지역 레이어 추가
            Map.addLayer(Data, {'color': 'orange'}, 'Presence')
            
            Map.centerObject(Data.geometry(), 9)
            Map.add_child(folium.LatLngPopup())
                                                    
            # 지도에 레이어 컨트롤 추가
            Map.add_child(folium.LayerControl())
            # Draw 플러그인을 추가하여 사용자가 영역을 선택할 수 있도록 합니다.
            draw = Draw(
                export=True
            )
            draw.add_to(Map)

            # folium 지도 객체를 HTML 문자열로 변환하여 Streamlit에 표시합니다.
            map_html = Map._repr_html_()
            components.html(map_html, height=800)


            st.subheader('산사태 발생 요인 분석')
            
            with st.form('latlon'):
                # change.display_map_with_draw(Map)
                lat = st.number_input("위도", value=35.95, format="%.5f")
                lon = st.number_input("경도", value=128.25, format="%.5f")

                # Every form must have a submit button.
                latlon_submitted = st.form_submit_button("Submit")

        if latlon_submitted:
            with st.spinner('산사태 발생 요인을 분석하는 중입니다! 잠시만 기다려 주세요.'):
                # 사용자 입력에 따라 데이터 처리 및 결과 표시
                change.process_user_input(lat, lon, variables)
                
                # 변수 설명#################################################################################
                variable_descriptions = {
                "crops": "경작지로 덮여있을 확률(0~1)",
                "water": "수역으로 덮여있을 확률(0~1)",
                "trees": "수목으로 덮여있을 확률(0~1)",
                "built": "주거·상업·공업지역으로 덮여있을 확률(0~1)",
                "grass": "초지로 덮여있을 확률(0~1)",
                "bare": "나지(맨땅)로 덮여있을 확률(0~1)",
                "aspect": "지형의 경사면이 향하는 방향(동서남북)",
                "elevation": "해수면으로부터의 높이를 m 단위로 측정",
                "slope": "지표면의 기울기",
                "susm": "mm 단위의 지하 토양 수분량",
                "VH": "Sentinel-1호(SAR) 위성에서 수직(V) 신호로 방출되고 지구 표면으로부터 수평(H) 신호로 수신되는 신호 방향",
                "VV": "Sentinel-1호(SAR) 위성에서 수직(V) 신호로 방출되고 지구 표면으로부터 수직(V) 신호로 수신되는 신호 방향"
                }

                def generate_variable_descriptions(variables, variable_descriptions):
                # 선택된 변수에 대한 설명을 Markdown 형식의 문자열로 생성합니다.
                    markdown_text = ""
                    for var in variables:
                        description = variable_descriptions.get(var, "")
                        if description:
                            markdown_text += f"- **{var} ({description})**\n"
                    return markdown_text

                # 선택된 권역에 따라 변수 이름이 포함된 'variables' 리스트를 가정합니다.
                markdown_text = generate_variable_descriptions(variables, variable_descriptions)

                # Markdown 텍스트를 두 개의 칼럼으로 나눕니다.
                half = len(markdown_text.split('\n')) // 2
                col1_text = '\n'.join(markdown_text.split('\n')[:half])
                col2_text = '\n'.join(markdown_text.split('\n')[half:])
                    
                st.write('<사용된 변수 설명>')
                with st.container():
                    col1, col2 = st.columns(2)
                    
                with col1:
                    st.markdown(col1_text)
                with col2:
                    st.markdown(col2_text)

                # feature_importance 이미지를 표시합니다.#########################################################
                # 그래프 코드
                import matplotlib.pyplot as plt
                import seaborn as sns
                sorted_pairs = sorted(zip(importance, variables), reverse=True)
                sorted_importance, sorted_variables = zip(*sorted_pairs)
                colors = sns.color_palette('rocket', len(sorted_variables))
                plt.figure(figsize=(10, 5))
                bars = plt.barh(sorted_variables, sorted_importance, color=colors)
                plt.xlabel('Importance')
                plt.title('Feature Importance')
                for bar in bars:
                    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', va='center')
                plt.gca().invert_yaxis()

                # Streamlit에서 그래프 표시
                st.pyplot(plt)

if __name__ == "__main__":
    app()
