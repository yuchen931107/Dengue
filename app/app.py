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
    
    # 確保資料裡的 Town 名稱沒有多餘的空白或「台南市」字眼，以利後續與地圖精準核對
    df['Town'] = df['Town'].astype(str).str.replace("台南市", "").str.replace("臺南市", "").str.strip()
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("找不到 data.csv 檔案，請確認檔案位置。")
    st.stop()


# 2. 載入台南市 GeoJSON 地理邊界 (透過快取加速，只會下載一次)
@st.cache_data
def load_geojson():
    # 讀取開源社群提供的台灣鄉鎮輕量版 GeoJSON
    url = "https://raw.githubusercontent.com/ronnywang/twgeojson/master/twtown2010.3.json"
    response = requests.get(url)
    geojson = response.json()
    
    # 【關鍵過濾】只保留「臺南市」的區域，避免畫到台中市或嘉義市的「東區」！
    tainan_features = []
    for feature in geojson['features']:
        # 依照原始地圖資料的屬性 COUNTYNAME 來篩選
        if feature['properties'].get('COUNTYNAME') == '臺南市':
            tainan_features.append(feature)
            
    # 把過濾好的台南區塊塞回 GeoJSON 格式中
    geojson['features'] = tainan_features
    return geojson

# 取得台南地圖
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
# 主畫面：面量圖 (Choropleth Map) 渲染
# ==========================================
st.title("🗺️ 台南市登革熱區塊分佈圖")
st.write(f"目前顯示：**{selected_year} 年 第 {selected_week} 週**的 `{heat_metric}` 分布狀況")

mask = (df['Year'] == selected_year) & (df['Week'] == selected_week)
df_filtered = df[mask]

if df_filtered.empty or df_filtered[heat_metric].sum() == 0:
    st.info("這一週沒有任何數據或數值皆為零，無法繪製地圖。請嘗試拖動左側的週次滑桿！")
else:
    # 畫出行政區塊的面量圖 (Choropleth)
    fig = px.choropleth_mapbox(
        df_filtered,
        geojson=tainan_geojson,             # 給予我們剛剛過濾好的台南市邊界
        locations='Town',                   # data.csv 裡面用來對應的欄位 (例如 "東區")
        featureidkey='properties.TOWNNAME', # GeoJSON 裡面用來對應的屬性 (例如 "東區")
        color=heat_metric,                  # 決定顏色的數值欄位
        color_continuous_scale="Reds",      # 紅色系漸層
        mapbox_style="carto-positron",      # 乾淨的底圖，凸顯疫情數據
        zoom=9.5,                           # 預設縮放大小
        center={"lat": 23.15, "lon": 120.25}, # 視角對準台南市
        opacity=0.7                         # 設定區塊透明度 0.7，這樣才看得到底下的街道圖
    )
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
with st.expander("點擊查看當週地圖對應的原始資料"):
    st.dataframe(df_filtered[['Town', 'Year', 'Week', heat_metric]])
