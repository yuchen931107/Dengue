import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="台南登革熱區塊分佈圖", layout="wide")

# 1. 讀取與清理資料
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data.csv")
    df = pd.read_csv(file_path)
    df['Town'] = df['Town'].astype(str).str.replace("台南市", "").str.replace("臺南市", "").str.strip()
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("找不到 data.csv 檔案，請確認檔案位置。")
    st.stop()


# 2. 載入台南市 GeoJSON 地理邊界 (終極修復版：處理縣市合併的字尾問題)
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/ronnywang/twgeojson/master/twtown2010.3.json"
    response = requests.get(url)
    geojson = response.json()
    
    tainan_features = []
    for feature in geojson['features']:
        props = feature.get('properties', {})
        county = props.get('COUNTYNAME', '')
        
        # 抓出 2010 年的「臺南縣」與「臺南市」
        if county in ['臺南市', '台南市', '臺南縣', '台南縣']:
            town_name = props.get('TOWNNAME', '').strip()
            
            # 【歷史遺毒終極修復】將舊制的「鄉」、「鎮」、「市」全部統一替換成「區」
            if town_name.endswith('鄉') or town_name.endswith('鎮') or town_name.endswith('市'):
                town_name = town_name[:-1] + '區'
            
            # 強制賦予每個區塊一個 ID
            feature['id'] = town_name
            tainan_features.append(feature)
            
    geojson['features'] = tainan_features
    return geojson

tainan_geojson = load_geojson()


# ==========================================
# 側邊欄：互動式控制面板
# ==========================================
st.sidebar.title("⚙️ 面量圖控制面板")

heat_metric = st.sidebar.selectbox(
    "選擇觀察指標", 
    options=['Case_Count', 'RT_level', 'BI', 'CI', 'HI']
)

available_years = sorted(df['Year'].unique())
selected_year = st.sidebar.selectbox("選擇年份", available_years)

available_weeks = sorted(df[df['Year'] == selected_year]['Week'].unique())
selected_week = st.sidebar.select_slider(
    "調整週次(日期)觀察疫情變化", 
    options=available_weeks,
    value=available_weeks[0]
)


# ==========================================
# 主畫面：純區塊圖 (Choropleth) 渲染
# ==========================================
st.title("🗺️ 台南市登革熱區塊分佈圖")

mask = (df['Year'] == selected_year) & (df['Week'] == selected_week)
df_filtered = df[mask]

if df_filtered.empty or df_filtered[heat_metric].sum() == 0:
    st.info("這一週沒有任何數據或數值皆為零，無法繪製地圖。請嘗試拖動左側的週次滑桿！")
else:
    # 畫出行政區塊的面量圖 (純幾何拼圖)
    fig = px.choropleth(
        df_filtered,
        geojson=tainan_geojson,
        locations='Town',                   # 對應 df 的 'Town' (例如 '七股區')
        color=heat_metric,
        color_continuous_scale="Reds",      # 紅色系漸層
        hover_name='Town'                   # 游標移過去顯示名稱
    )
    
    # 隱藏地球背景、完美縮放，並加上麥卡托投影 (避免台灣形狀被扭曲變胖)
    fig.update_geos(
        fitbounds="locations", 
        visible=False,
        projection_type="mercator"
    )
    
    # 加上白色邊框，增加各行政區拼圖的立體質感
    fig.update_traces(marker_line_width=1.5, marker_line_color="white")
    
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        plot_bgcolor='rgba(0,0,0,0)',       # 將圖表背景設為完全透明
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
with st.expander("點擊查看當週地圖對應的原始資料"):
    st.dataframe(df_filtered[['Town', 'Year', 'Week', heat_metric]])
