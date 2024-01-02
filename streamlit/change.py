import ee 
import geemap
import folium 
import pandas as pd 
from scipy.stats import chi2
import streamlit as st
import streamlit.components.v1 as components
import time
from folium.plugins import Draw, Fullscreen
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import os
import parks
from datetime import datetime, timedelta
import json
import geopandas as gpd

national_parks = parks.get_parks()
park_ids = parks.get_ids()

# V-World 타일 서비스 URL (API 키 포함)
vworld_satellite_url = "http://api.vworld.kr/req/wmts/1.0.0/{api_key}/Satellite/{z}/{y}/{x}.jpeg"
vworld_hybrid_url = "http://api.vworld.kr/req/wmts/1.0.0/{api_key}/Hybrid/{z}/{y}/{x}.png"

# API 키를 여기에 입력하세요
api_key = "DCFAECAC-2343-3CB2-81AA-1FE195545C28"

def add_ee_layer(self, ee_image_object, vis_params, name):
    """Adds Earth Engine layers to a folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles = map_id_dict['tile_fetcher'].url_format,
        attr = 'Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name = name,
        overlay = True,
        control = True).add_to(self)


def display_map_with_draw(m):
    # V-World Satellite 레이어를 기본 레이어로 추가
    folium.TileLayer(
        tiles=vworld_satellite_url.replace("{api_key}", api_key),
        attr='V-World Satellite',
        name='V-World Satellite',
        overlay=False
    ).add_to(m)

    # V-World Hybrid 레이어 추가
    folium.TileLayer(
        tiles=vworld_hybrid_url.replace("{api_key}", api_key),
        attr='V-World Hybrid',
        name='V-World Hybrid',
        overlay=True
    ).add_to(m)

    # 기본 Folium 레이어 (예: OpenStreetMap)도 추가
    folium.TileLayer('OpenStreetMap').add_to(m)

    # Draw 플러그인을 추가하여 사용자가 영역을 선택할 수 있도록 합니다.
    draw = Draw(
        draw_options={
            'polyline': False,  # Polyline 그리기 허용
            'polygon': True,   # Polygon 그리기 허용
            'rectangle': True, # Rectangle 그리기 허용
            'circle': False,   # Circle 그리기 비허용
            'marker': False,   # Marker 그리기 비허용
            'circlemarker': False, # CircleMarker 그리기 비허용
        },
        edit_options={
            'poly': {
                'allowIntersection': False  # 교차 허용 여부 설정
            }
        },
        export=True
    )
    draw.add_to(m)

    # 전체화면 버튼 추가
    Fullscreen().add_to(m)
    
    m.add_child(folium.LatLngPopup())
                                            
    # 지도에 레이어 컨트롤 추가
    m.add_child(folium.LayerControl())

    # folium 지도 객체를 HTML 문자열로 변환하여 Streamlit에 표시합니다.
    map_html = m._repr_html_()
    components.html(map_html, height=800)

# 누적분포함수를 계산
def chi2cdf(chi2, df):
    """Calculates Chi square cumulative distribution function for
    df degrees of freedom using the built-in incomplete gamma
    function gammainc().
    """
    return ee.Image(chi2.divide(2)).gammainc(ee.Number(df).divide(2))

# 공분산 행렬식 계산
def det(im):
    """Calculates determinant of 2x2 diagonal covariance matrix."""
    return im.expression('b(0)*b(1)')

# im_list 내 첫 j개 이미지의 합의 행렬식 로그를 반환하는 함수
def log_det_sum(im_list, j):
    """Returns log of determinant of the sum of the first j images in im_list."""
    im_ist = ee.List(im_list)
    sumj = ee.ImageCollection(im_list.slice(0, j)).reduce(ee.Reducer.sum())
    return ee.Image(det(sumj)).log()

# im_list의 j번째 이미지의 행렬식 로그를 반환하는 함수
def log_det(im_list, j):
    """Returns log of the determinant of the jth image in im_list."""
    im = ee.Image(ee.List(im_list).get(j.subtract(1)))
    return ee.Image(det(im)).log()

#  im_list에 대해 -2logRj를 계산하고 P값과 -2logRj를 반환하는 함수
def pval(im_list, j, m = 5):
    """Calculates -2logRj for im_list and returns P value and -2logRj."""
    im_list = ee.List(im_list)
    j = ee.Number(j)
    m2logRj = (log_det_sum(im_list, j.subtract(1))
            .multiply(j.subtract(1))
            .add(log_det(im_list, j))
            .add(ee.Number(2).multiply(j).multiply(j.log()))
            .subtract(ee.Number(2).multiply(j.subtract(1))
            .multiply(j.subtract(1).log()))
            .subtract(log_det_sum(im_list,j).multiply(j))
            .multiply(-2).multiply(m))
    pv = ee.Image.constant(1).subtract(chi2cdf(m2logRj, 2))
    return (pv, m2logRj)

# 이미지 리스트에 대한 P값 배열을 사전 계산하는 함수.
def p_values(im_list):
    """Pre-calculates the P-value array for a list of images."""
    im_list = ee.List(im_list)
    k = im_list.length()

    #  k와 j의 조합에 대해 pval 계산을 정리하는 함수
    def ells_map(ell):
        """Arranges calculation of pval for combinations of k and j."""
        ell = ee.Number(ell)
        # k-l+1부터 k까지의 시리즈를 슬라이스 (이미지 인덱스는 0부터 시작)
        im_list_ell = im_list.slice(k.subtract(ell), k)

        #  k와 j의 조합에 대해 pval 계산을 적용하는 함수
        def js_map(j):
            """Applies pval calculation for combinations of k and j."""
            j = ee.Number(j)
            pv1, m2logRj1 = pval(im_list_ell, j)
            return ee.Feature(None, {'pv': pv1, 'm2logRj': m2logRj1})

        # j=2,3,...,l에 대해 매핑
        js = ee.List.sequence(2, ell)
        pv_m2logRj = ee.FeatureCollection(js.map(js_map))
        # m2logRj 이미지의 컬렉션에서 m2logQl 계산
        m2logQl = ee.ImageCollection(pv_m2logRj.aggregate_array('m2logRj')).sum()
        pvQl = ee.Image.constant(1).subtract(chi2cdf(m2logQl, ell.subtract(1).multiply(2)))
        pvs = ee.List(pv_m2logRj.aggregate_array('pv')).add(pvQl)
        return pvs

    # l = k,...,2에 대해 매핑
    ells = ee.List.sequence(k, 2, -1)
    pv_arr = ells.map(ells_map)
    # P 값 배열을 반환 (ell = k,...,2, j = 2,...,l)
    return pv_arr

# 변화지도를 계산하는 함수; pv_arr의 j 인덱스에 대해 반복
# current: 현재 P값
# prev: 이전 단계의 결과
def filter_j(current, prev):
    """Calculates change maps; iterates over j indices of pv_arr."""
    pv = ee.Image(current)
    prev = ee.Dictionary(prev)
    pvQ = ee.Image(prev.get('pvQ'))
    i = ee.Number(prev.get('i'))

    cmap = ee.Image(prev.get('cmap'))
    smap = ee.Image(prev.get('smap'))
    fmap = ee.Image(prev.get('fmap'))
    bmap = ee.Image(prev.get('bmap'))
    alpha = ee.Image(prev.get('alpha'))

    j = ee.Number(prev.get('j'))

    cmapj = cmap.multiply(0).add(i.add(j).subtract(1))
    
    # Check      Rj?            Ql?                  Row i?
    tst = pv.lt(alpha).And(pvQ.lt(alpha)).And(cmap.eq(i.subtract(1)))

    # Then update cmap...
    cmap = cmap.where(tst, cmapj)

    # ...and fmap...
    fmap = fmap.where(tst, fmap.add(1))

    # ...and smap only if in first row.
    smap = ee.Algorithms.If(i.eq(1), smap.where(tst, cmapj), smap)

    # bmap 밴드 생성 및 bmap 이미지에 추가
    idx = i.add(j).subtract(2)
    tmp = bmap.select(idx)
    bname = bmap.bandNames().get(idx)
    tmp = tmp.where(tst, 1)
    tmp = tmp.rename([bname])
    bmap = bmap.addBands(tmp, [bname], True)
    return ee.Dictionary({'i': i, 'j': j.add(1), 'alpha': alpha, 'pvQ': pvQ,
                        'cmap': cmap, 'smap': smap, 'fmap': fmap, 'bmap':bmap})

# 변화지도 계산을 준비하는 함수; pv_arr의 행 인덱스에 대해 반복.
def filter_i(current, prev):
    """Arranges calculation of change maps; iterates over row-indices of pv_arr."""
    current = ee.List(current)
    pvs = current.slice(0, -1 )
    pvQ = ee.Image(current.get(-1))
    prev = ee.Dictionary(prev)
    i = ee.Number(prev.get('i'))
    alpha = ee.Image(prev.get('alpha'))
    median = prev.get('median')

    # 필요한 경우 Ql P값 필터링
    pvQ = ee.Algorithms.If(median, pvQ.focalMedian(2.5), pvQ)
    cmap = prev.get('cmap')
    smap = prev.get('smap')
    fmap = prev.get('fmap')
    bmap = prev.get('bmap')
    first = ee.Dictionary({'i': i, 'j': 1, 'alpha': alpha ,'pvQ': pvQ,
                        'cmap': cmap, 'smap': smap, 'fmap': fmap, 'bmap': bmap})
    result = ee.Dictionary(ee.List(pvs).iterate(filter_j, first))
    return ee.Dictionary({'i': i.add(1), 'alpha': alpha, 'median': median,
                        'cmap': result.get('cmap'), 'smap': result.get('smap'),
                        'fmap': result.get('fmap'), 'bmap': result.get('bmap')})

# 변화 방향을 계산하는 반복 함수
def dmap_iter(current, prev):
    """변화 방향에 따라 값 재분류"""
    prev = ee.Dictionary(prev)
    j = ee.Number(prev.get('j'))
    image = ee.Image(current)
    avimg = ee.Image(prev.get('avimg'))
    diff = image.subtract(avimg)

    # Get positive/negative definiteness.
    posd = ee.Image(diff.select(0).gt(0).And(det(diff).gt(0)))
    negd = ee.Image(diff.select(0).lt(0).And(det(diff).gt(0)))
    bmap = ee.Image(prev.get('bmap'))
    bmapj = bmap.select(j)
    dmap = ee.Image.constant(ee.List.sequence(1, 3))
    bmapj = bmapj.where(bmapj, dmap.select(2))
    bmapj = bmapj.where(bmapj.And(posd), dmap.select(0))
    bmapj = bmapj.where(bmapj.And(negd), dmap.select(1))
    bmap = bmap.addBands(bmapj, overwrite=True)

    # Update avimg with provisional means.
    i = ee.Image(prev.get('i')).add(1)
    avimg = avimg.add(image.subtract(avimg).divide(i))

    # 변화가 발생한 경우 avimg를 현재 이미지로 설정하고 i를 1로 재설정
    avimg = avimg.where(bmapj, image)
    i = i.where(bmapj, 1)
    return ee.Dictionary({'avimg': avimg, 'bmap': bmap, 'j': j.add(1), 'i': i})

def change_maps(im_list, median=False, alpha=0.01):
    """Calculates thematic change maps."""
    k = im_list.length()

    # Pre-calculate the P value array.
    pv_arr = ee.List(p_values(im_list))

    # Filter P values for change maps.
    cmap = ee.Image(im_list.get(0)).select(0).multiply(0)
    bmap = ee.Image.constant(ee.List.repeat(0,k.subtract(1))).add(cmap)
    alpha = ee.Image.constant(alpha)
    first = ee.Dictionary({'i': 1, 'alpha': alpha, 'median': median,
                        'cmap': cmap, 'smap': cmap, 'fmap': cmap, 'bmap': bmap})
    result = ee.Dictionary(pv_arr.iterate(filter_i, first))

    # Post-process bmap for change direction.
    bmap =  ee.Image(result.get('bmap'))
    avimg = ee.Image(im_list.get(0))
    j = ee.Number(0)
    i = ee.Image.constant(1)
    first = ee.Dictionary({'avimg': avimg, 'bmap': bmap, 'j': j, 'i': i})
    dmap = ee.Dictionary(im_list.slice(1).iterate(dmap_iter, first)).get('bmap')
    return ee.Dictionary(result.set('bmap', dmap))


def get_aoi(park_name=None, geojson=None):
    # 관심 국립공원 경계 추출
    def get_boundary(park_name=None, geojson=None):
        if park_name is not None:
            selected_park = national_parks.loc[national_parks['park_ko'] == park_name, 'park_en'].item()

            if selected_park == "Jirisan":
                return ee.FeatureCollection("WCMC/WDPA/current/polygons").filter(ee.Filter.eq("WDPAID", 767))
            elif selected_park == "Seoraksan":
                return ee.FeatureCollection("WCMC/WDPA/current/polygons").filter(ee.Filter.eq("WDPAID", 768))
            else:
                # 다른 국립공원들은 이름으로 필터링
                return ee.FeatureCollection("WCMC/WDPA/current/polygons").filter(ee.Filter.eq("NAME", selected_park))

        elif geojson is not None:
            # GeoJSON 파일을 읽고 Earth Engine의 FeatureCollection으로 변환
            aoi_geojson = json.load(geojson)
            return ee.FeatureCollection(aoi_geojson)  

    return get_boundary(park_name, geojson)

def get_polygon(lat, lon):
    poi = ee.Geometry.Point([lon, lat])

    # 중심점을 기반으로 하는 10km 크기의 정사각형 폴리곤 생성
    side_length = 10 * 1000  # 10km
    aoi_sub = ee.Geometry.Point([lon, lat]).buffer(side_length / 2).bounds()

    return aoi_sub

# 식생지수 계산 함수
def calculate_vegetation_index(image_collection, aoi):
    # 이미지 컬렉션에서 식생지수를 계산하는 함수를 적용합니다.
    def add_vegetation_index(image):
        vv = image.select('VV')
        vh = image.select('VH')
        vegetation_index = vh.multiply(4).divide(vv.add(vh))
        return image.addBands(vegetation_index.rename('RVI'))
    return image_collection.map(add_vegetation_index)

# 식생지수 통계 추출 함수
def extract_statistics(image_collection, aoi):
    # 이미지 컬렉션에서 각 이미지의 식생지수 평균을 추출합니다.
    def get_stats(image):
        mean_dict = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30
        ).get('RVI')
        return image.set('date', image.date().format()).set('mean_rvi', mean_dict)
    return image_collection.map(get_stats)

# 식생지수 통계를 데이터프레임으로 변환하는 함수
def stats_to_dataframe(stats_list):
    df = pd.DataFrame(stats_list, columns=['Date', 'Mean_RVI'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns={'Date': 'ds', 'Mean_RVI': 'y'}, inplace=True)
    return df

def get_soil_prop(param):
    """
    이 함수는 토양 속성 이미지를 반환합니다.
    param (str): 다음 중 하나여야 합니다:
        "sand"     - 모래 함량
        "clay"     - 점토 함량
        "orgc"     - 유기 탄소 함량
    """
    if param == "sand":  # 모래 함량 [%w]
        snippet = "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02"
        # 데이터셋 설명에 따른 스케일 팩터 정의.
        scale_factor = 1 * 0.01
    elif param == "clay":  # 점토 함량 [%w]
        snippet = "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"
        # 데이터셋 설명에 따른 스케일 팩터 정의.
        scale_factor = 1 * 0.01
    elif param == "orgc":  # 유기 탄소 함량 [g/kg]
        snippet = "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02"
        # 데이터셋 설명에 따른 스케일 팩터 정의.
        scale_factor = 5 * 0.001  # kg/kg로 변환하기 위함.
    else:
        return print("error")
    # 스케일 팩터를 ee.Image에 적용합니다.
    dataset = ee.Image(snippet).multiply(scale_factor)
    return dataset

def sum_resampler(coll, freq, unit, scale_factor, band_name):
    """
    이 함수는 ee.ImageCollection의 시간 규모를 재샘플링하는 것을 목표로 합니다.
    함수는 선택된 빈도에 대한 밴드의 평균 합계를 가진 ee.ImageCollection을 반환합니다.
    coll: (ee.ImageCollection) 한 밴드만 처리 가능
    freq: (int) 재샘플링 빈도에 해당
    unit: (str) 재샘플링 시간 단위. 'day', 'month', 'year'이어야 함
    scale_factor (float): 적절한 단위로 값을 얻기 위한 스케일링 인자
    band_name (str) 출력 밴드의 이름
    """
    # 컬렉션의 초기 및 최종 날짜 정의.
    firstdate = ee.Date(
        coll.sort("system:time_start", True).first().get("system:time_start")
    )
    lastdate = ee.Date(
        coll.sort("system:time_start", False).first().get("system:time_start")
    )
    # 두 날짜 사이의 시간 차이 계산.
    diff_dates = lastdate.difference(firstdate, unit)

    # 새로운 시간 인덱스(출력용) 정의.
    new_index = ee.List.sequence(0, ee.Number(diff_dates), freq)

    # 새 시간 인덱스에 적용할 함수 정의.
    def apply_resampling(date_index):
        # 고려할 시작 날짜 정의.
        startdate = firstdate.advance(ee.Number(date_index), unit)

        # 원하는 빈도에 따라 고려할 종료 날짜 정의.
        enddate = startdate.advance(freq, unit)

        # 시작 및 종료 날짜 사이의 일수 계산.
        diff_days = enddate.difference(startdate, "day")

        # 합성 이미지 계산.
        image = (
            coll.filterDate(startdate, enddate)
            .mean()
            .multiply(diff_days)
            .multiply(scale_factor)
            .rename(band_name)
        )
        # 적절한 시간 인덱스를 가진 최종 이미지 반환.
        return image.set("system:time_start", startdate.millis())

    # 새 시간 인덱스에 함수 매핑.
    res = new_index.map(apply_resampling)

    # 결과를 ee.ImageCollection으로 변환.
    res = ee.ImageCollection(res)
    return res        

# 깊이에 따른 토양 속성의 평균 값을 계산하는 함수.
def olm_prop_mean(olm_image, band_output_name):
    """
    식물이 뿌리를 통해 접근할 수 있는 토양 깊이에 걸쳐 있는 수분 함량
    """
    mean_image = olm_image.expression(
        "(b0 + b10 + b30 + b60 + b100 + b200) / 6",
        {
            "b0": olm_image.select("b0"),
            "b10": olm_image.select("b10"),
            "b30": olm_image.select("b30"),
            "b60": olm_image.select("b60"),
            "b100": olm_image.select("b100"),
            "b200": olm_image.select("b200"),
        },
    ).rename(band_output_name)
    return mean_image

def call_recharge_calculator(image_coll1, image_coll2, fcm, wpm, stfc):

    def recharge_calculator(image, image_list):

        """
        각 반복마다 수행되는 연산들을 포함합니다.
        """
        # 현재 ee.Image 컬렉션의 날짜 결정.
        localdate = image.date().millis()

        # 리스트에 저장된 이전 이미지 가져오기.
        prev_im = ee.Image(ee.List(image_list).get(-1))

        # 이전 APWL과 ST 가져오기.
        prev_apwl = prev_im.select("apwl")
        prev_st = prev_im.select("st")

        # 현재 강수량과 증발산량 가져오기.
        pr_im = image.select("pr")
        pet_im = image.select("pet")

        # 재충전, APWL, ST와 관련된 새로운 밴드 초기화
        new_rech = (
            ee.Image(0)
            .set("system:time_start", localdate)
            .select([0], ["rech"])
            .float()
        )

        new_apwl = (
            ee.Image(0)
            .set("system:time_start", localdate)
            .select([0], ["apwl"])
            .float()
        )

        new_st = (
            prev_st.set("system:time_start", localdate).select([0], ["st"]).float()
        )

        # 조건에 따라 밴드 계산 수행
        # CASE 1: PET > P 일 경우 (증발산량 > 강수량)
        zone1 = pet_im.gt(pr_im)

        # zone1에서 APWL(누적 잠재적 수분 손실) 계산
        zone1_apwl = prev_apwl.add(pet_im.subtract(pr_im)).rename("apwl")

        # zone1의 APWL 값 구현
        new_apwl = new_apwl.where(zone1, zone1_apwl)

        # zone1에서 ST(저장 수분) 계산
        zone1_st = prev_st.multiply(
            ee.Image.exp(zone1_apwl.divide(stfc).multiply(-1))
        ).rename("st")

        # zone1의 ST 값 구현
        new_st = new_st.where(zone1, zone1_st)

        # CASE 2: PET <= P 일 경우 (증발산량 <= 강수량)
        zone2 = pet_im.lte(pr_im)

        # zone2에서 ST 계산.
        zone2_st = prev_st.add(pr_im).subtract(pet_im).rename("st")

        # zone2의 ST 값 구현.
        new_st = new_st.where(zone2, zone2_st)

        # CASE 2.1: PET <= P 및 ST >= STfc 일 경우
        zone21 = zone2.And(zone2_st.gte(stfc))

        # zone21에서 재충전 계산 즉, 필드 용량 이상의 수분이 있을때, 재충전량 계산
        zone21_re = zone2_st.subtract(stfc).rename("rech")

        # zone21의 재충전 값 구현.
        new_rech = new_rech.where(zone21, zone21_re)

        # zone21의 ST 값 구현.
        new_st = new_st.where(zone21, stfc)

        # CASE 2.2: PET <= P 및 ST < STfc 일 경우 즉, 필드 용량 미만의 수분이 있을때
        zone22 = zone2.And(zone2_st.lt(stfc))

        # zone22에서 APWL 계산.
        zone22_apwl = (
            stfc.multiply(-1).multiply(ee.Image.log(zone2_st.divide(stfc))).rename("apwl")
        )

        # zone22의 APWL 값 구현.
        new_apwl = new_apwl.where(zone22, zone22_apwl)

        # 재충전 계산이 가능한 영역 주위에 마스크 생성.
        mask = pet_im.gte(0).And(pr_im.gte(0)).And(fcm.gte(0)).And(wpm.gte(0))

        # 마스크 적용.
        new_rech = new_rech.updateMask(mask)

        # 모든 밴드를 새로운 ee.Image에 추가.
        new_image = new_rech.addBands(ee.Image([new_apwl, new_st, pr_im, pet_im]))

        # 새로운 ee.Image를 ee.List에 추가.
        return ee.List(image_list).add(new_image)
    return image_coll1.iterate(recharge_calculator, image_coll2)

def local_profile(dataset, poi, buffer, olm_bands):
    """
    buffer: 샘플링 범위 (미터 단위)
    """
    # 관심 지역에서 속성을 얻습니다.
    prop = dataset.sample(poi, buffer).select(olm_bands).getInfo()

    # 관심 있는 특성을 선택합니다.
    profile = prop["features"][0]["properties"]

    # 딕셔너리를 재구성합니다.
    profile = {key: round(val, 3) for key, val in profile.items()}

    return profile

def ee_array_to_df(arr, list_of_bands):
    """
    클라이언트 측의 ee.Image.getRegion 배열을 pandas.DataFrame으로 변환합니다.
    arr: getRegion 메소드로부터 얻은 데이터 배열
    list_of_bands: 데이터프레임에 포함시킬 밴드 목록
    """
    # 데이터 배열을 데이터프레임으로 변환.
    df = pd.DataFrame(arr)

    # 헤더 재배열.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # 데이터를 숫자 값으로 변환.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors="coerce")

    # 'time' 필드를 datetime으로 변환.
    df["datetime"] = pd.to_datetime(df["time"], unit="ms")

    # 관심 있는 컬럼만 유지.
    df = df[["time", "datetime", *list_of_bands]]

    # 'datetime' 컬럼을 인덱스로 설정.
    df = df.set_index("datetime")
    return df

def get_local_recharge(rech_coll, i_date, f_date, lon, lat, scale):
    """
    주어진 기간 동안 지정된 위치에서 월별 누적 지하수 재충전량을 설명하는 pandas DataFrame을 반환합니다.
    """
    # 관심 지역의 경계 정의
    poi = ee.Geometry.Point([lon, lat])
    # 관심 지역 주변의 재충전을 평가합니다.
    rarr = rech_coll.filterDate(i_date, f_date).getRegion(poi, scale).getInfo()
    # 결과를 pandas 데이터프레임으로 변환합니다.
    rdf = ee_array_to_df(rarr, ["pr", "pet", "apwl", "st", "rech"]).sort_index()
    return rdf

# 국립공원 경계를 지도에 추가하는 함수 정의
def add_park_boundaries(my_map, park_name):
    for park_name in national_parks:
        # 지리산과 설악산은 WDPAID를 사용하여 경계를 검색합니다.
        if park_name in park_ids:
            park_boundary = ee.FeatureCollection("WCMC/WDPA/current/polygons") \
                .filter(ee.Filter.eq("WDPAID", park_ids[park_name]))
        else:
            park_boundary = ee.FeatureCollection("WCMC/WDPA/current/polygons") \
                .filter(ee.Filter.eq("NAME", park_name))
        
        # EE 결과를 GeoJson으로 변환
        geojson = geemap.ee_to_geojson(park_boundary)
        
        # folium.GeoJson을 사용하여 지도에 경계 추가
        folium.GeoJson(
            geojson,
            name=park_name,
            style_function=lambda feature: {
                'color': 'red',
                'weight': 1,
                'fillOpacity': 0
            }
        ).add_to(my_map)
        
####### 권역별 파일 이름 및 임계값 설정 ##############################
#######그래프 그리기 위한 변수와 변수 중요도 표시#####################
####### 예측 정확도값 표시############################################

def get_region_data(region):
    if region == '수도권':
        model_average_file = '.\\권역별_pkl\\ModelAverage_수도권ver1.pkl'
        data_file = '.\\권역별_pkl\\AcuallyData_수도권ver1.pkl'
        threshold = 0.4416666666666666
        variables = ['bare', 'crops', 'VV', 'built', 'water', 'slope', 'aspect', 'elevation']
        importance = [2.72, 5.53, 5.78, 6.13, 6.71, 18.26, 21.09, 28.88]
        auc_roc = 0.868002
        auc_pr = 0.842519

    elif region == '영남권':
        model_average_file = '.\\권역별_pkl\\ModelAverage_영남권ver1.pkl'
        data_file = '.\\권역별_pkl\\AcuallyData_영남권ver1.pkl'
        threshold = 0.5083333333333333
        variables = ['water', 'aspect', 'built', 'trees', 'crops', 'slope', 'elevation']
        importance = [9.82, 10.88, 11.67, 15.44, 24.36, 29.39, 39.43]
        auc_roc = 0.785778
        auc_pr = 0.734642

    elif region == '호남권':
        model_average_file = '.\\권역별_pkl\\ModelAverage_호남권ver1.pkl'
        data_file = '.\\권역별_pkl\\AcuallyData_호남권ver1.pkl'
        threshold = 0.6083333333333332
        variables = ['water', 'susm', 'aspect', 'built', 'slope', 'trees', 'crops', 'elevation']
        importance = [5.13, 5.14, 5.96, 6.00, 10.54, 15.52, 20.95, 26.86]
        auc_roc = 0.860623
        auc_pr = 0.861038

    elif region == '충청권':
        model_average_file = '.\\권역별_pkl\\ModelAverage_충청권ver1.pkl'
        data_file = '.\\권역별_pkl\\AcuallyData_충청권ver1.pkl'
        threshold = 0.5166666666666666
        variables = ['aspect', 'grass', 'water', 'built', 'crops', 'slope', 'trees', 'elevation']
        importance = [2.65, 4.11, 4.85, 15.27, 15.43, 22.91, 28.53, 29.89]
        auc_roc = 0.858190
        auc_pr = 0.825055

    else:  # 강원권
        model_average_file = '.\\권역별_pkl\\ModelAverage_강원권ver1.pkl'
        data_file = '.\\권역별_pkl\\AcuallyData_강원권ver1.pkl'
        threshold = 0.4499999999999999
        variables = ['water', 'aspect', 'slope', 'trees', 'crops', 'built', 'elevation']
        importance = [2.11, 2.14, 3.28, 3.86, 5.40, 5.99, 11.15]
        auc_roc = 0.844258
        auc_pr = 0.805489

    return model_average_file, data_file, threshold, variables, importance, auc_roc, auc_pr

####### 변수 불러오기 ##################################################################################

# Sentinel-1 평균 값 가져오기
def get_sentinel_1_mean(aoi):
    start_date = '2014-01-01'
    end_date = '2022-12-31'

    s1_images = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterBounds(aoi) \
                .filterDate(ee.Date(start_date), ee.Date(end_date)) \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                .select(['VH','VV'])

    mean = s1_images.mean().reproject(crs='EPSG:4326', scale=30)
    mean_dict = mean.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=30
    )
    return mean_dict.getInfo()

# 최근 Sentinel-1 데이터 가져오기
def get_recent_sentinel_1_data(aoi, recent_date):
    s1_images = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterBounds(aoi) \
                .filterDate(ee.Date(recent_date), ee.Date(recent_date).advance(1, 'day')) \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                .select(['VH','VV'])

    recent_mean = s1_images.mean().reproject(crs='EPSG:4326', scale=30)
    recent_mean_dict = recent_mean.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=30
    )
    return recent_mean_dict.getInfo()

# SRTM 데이터 가져오기
def get_srtm_data(aoi):
    srtm = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(srtm)
    aspect = ee.Terrain.aspect(srtm)

    elevation = srtm.reduceRegion(reducer=ee.Reducer.first(), geometry=aoi, scale=30).get('elevation')
    slope_value = slope.reduceRegion(reducer=ee.Reducer.first(), geometry=aoi, scale=30).get('slope')
    aspect_value = aspect.reduceRegion(reducer=ee.Reducer.first(), geometry=aoi, scale=30).get('aspect')

    return elevation.getInfo(), slope_value.getInfo(), aspect_value.getInfo()

# Dynamic World 평균 값 가져오기 (수정됨)
def get_dynamic_world_data(aoi, date):
    nearest_date = find_nearest_date(aoi, date, 'GOOGLE/DYNAMICWORLD/V1')

    if nearest_date is None:
        return None, None, None, None, None, None

    start_date = nearest_date.strftime('%Y-%m-%d')
    end_date = (nearest_date + timedelta(days=1)).strftime('%Y-%m-%d')

    dw_images = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
                .filterBounds(aoi) \
                .filterDate(ee.Date(start_date), ee.Date(end_date)) \
                .select(['built', 'crops', 'water', 'trees', 'grass', 'bare'])

    mean_image = dw_images.mean().reproject(crs='EPSG:4326', scale=10)
    mean_values = mean_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10
    )

    built_mean = mean_values.get('built').getInfo()
    crops_mean = mean_values.get('crops').getInfo()
    water_mean = mean_values.get('water').getInfo()
    trees_mean = mean_values.get('trees').getInfo()
    grass_mean = mean_values.get('grass').getInfo()
    bare_mean = mean_values.get('bare').getInfo()

    return built_mean, crops_mean, water_mean, trees_mean, grass_mean, bare_mean


# SMAP 토양 수분 데이터 가져오기
def get_smap_soil_moisture(aoi):
    # 날짜 범위 설정
    start_date = '2014-01-01'
    end_date = '2022-12-31'

    # SMAP 토양 수분 데이터셋 필터링
    smap_data = ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture") \
                 .filterDate(ee.Date(start_date), ee.Date(end_date)) \
                 .select(['susm'])

    # 평균 이미지 계산
    mean_susm = smap_data.mean().reproject(crs='EPSG:4326', scale=10000) \
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=10000
                ).get('susm')

    # 결과 반환
    return mean_susm.getInfo()

# 가장 가까운 날짜 찾기
def find_nearest_date(aoi, target_date, collection_id):
    collection = ee.ImageCollection(collection_id) \
                .filterBounds(aoi)
    
    if collection_id == 'COPERNICUS/S1_GRD':
        collection = collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
                               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

    available_dates = collection.aggregate_array('system:time_start').getInfo()
    available_dates = [datetime.utcfromtimestamp(date / 1000) for date in available_dates]
    
    if not available_dates:
        return None
    
    nearest_date = min(available_dates, key=lambda d: abs(d - target_date))
    return nearest_date

# 사용자 입력 처리를 위한 함수
def process_user_input(lat, lon, variables):
    aoi = ee.Geometry.Point([lon, lat])
    current_date = datetime.now()

    # Sentinel-1 데이터셋에 대한 최근 날짜 찾기
    nearest_date_vh = find_nearest_date(aoi, current_date, 'COPERNICUS/S1_GRD')
    # Dynamic World 데이터셋에 대한 최근 날짜 찾기
    nearest_date_dw = find_nearest_date(aoi, current_date, 'GOOGLE/DYNAMICWORLD/V1')

    if nearest_date_vh is None or nearest_date_dw is None:
        st.write("가용한 위성 데이터가 없습니다.")
    else:
        # 지정된 기간 동안의 평균값
        period_mean_vh = get_sentinel_1_mean(aoi)
        period_built_mean, period_crops_mean, period_water_mean, period_trees_mean, period_grass_mean, period_bare_mean = get_dynamic_world_data(aoi, current_date)
        mean_susm = get_smap_soil_moisture(aoi)
        mean_elevation, mean_slope, mean_aspect = get_srtm_data(aoi)

        # 최근 값
        recent_mean_vh = get_recent_sentinel_1_data(aoi, nearest_date_vh)
        recent_built_mean, recent_crops_mean, recent_water_mean, recent_trees_mean, recent_grass_mean, recent_bare_mean = get_dynamic_world_data(aoi, nearest_date_dw)
        # 결과를 표 형태로 표시
  
# 과거 데이터 표시
    past_data = {
        'VH': [period_mean_vh.get('VH', None)],
        'VV': [period_mean_vh.get('VV', None)],
        'built': [period_built_mean],
        'crops': [period_crops_mean],
        'water': [period_water_mean],
        'trees': [period_trees_mean],
        'grass': [period_grass_mean],
        'bare': [period_bare_mean],
        'elevation': [mean_elevation],
        'slope': [mean_slope],
        'aspect': [mean_aspect],
        'susm': [mean_susm],
    }
    st.write("<과거(2014~2022) 평균적인 경향성 (사용된 변수)>")
    st.table({var: past_data[var] for var in variables})

    # 최근 데이터 표시
    recent_data = {
        '가장 가까운 날짜 (Sentinel-1)': [nearest_date_vh.strftime('%Y-%m-%d') if nearest_date_vh else 'None'],
        'VH': [recent_mean_vh.get('VH', None) if recent_mean_vh else 'None'],
        'VV': [recent_mean_vh.get('VV', None) if recent_mean_vh else 'None'],
        '가장 가까운 날짜 (Dynamic World)': [nearest_date_dw.strftime('%Y-%m-%d') if nearest_date_dw else 'None'],
        'built': [recent_built_mean if recent_built_mean else 'None'],
        'crops': [recent_crops_mean if recent_crops_mean else 'None'],
        'water': [recent_water_mean if recent_water_mean else 'None'],
        'trees': [recent_trees_mean if recent_trees_mean else 'None'],
        'grass': [recent_grass_mean if recent_grass_mean else 'None'],
        'bare': [recent_bare_mean if recent_bare_mean else 'None']
    }
    # 최근 데이터에서도 동일한 변수 사용
    st.write("<최근 값 (사용된 변수)>")
    # 가장 가까운 날짜 (Sentinel-1) 출력
    nearest_date_s1_str = nearest_date_vh.strftime('%Y-%m-%d') if nearest_date_vh else 'None'
    st.write('가장 가까운 날짜 (Sentinel-1): ' + nearest_date_s1_str)
    # 가장 가까운 날짜 (Dynamic World) 출력
    nearest_date_dw_str = nearest_date_dw.strftime('%Y-%m-%d') if nearest_date_dw else 'None'
    st.write('가장 가까운 날짜 (Dynamic World): ' + nearest_date_dw_str)
    st.table({var: recent_data[var] for var in variables if var in recent_data})

##########################################################################################################################

# 각 막대에 라벨을 붙이는 함수 정의.
def autolabel_soil_prop(rects, ax):
    """각 막대(*rects*) 위에 텍스트 라벨을 부착하여 높이를 표시합니다."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height) + "%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 수직 오프셋 3포인트.
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

# 각 막대에 레이블 추가 함수 정의
def autolabel_recharge(rects, ax):
    """각 막대 위에 높이를 표시하는 텍스트 레이블을 부착합니다."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(int(height)) + " mm",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 수직 오프셋 3 포인트
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
