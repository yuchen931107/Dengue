import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px

# 將網頁設定為寬螢幕模式 (更適合戰情室與雙地圖)
st.set_page_config(page_title="台南登革熱戰情儀表板", layout="wide")

# ==========================================
# 1. 讀取與清理資料
# ==========================================
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
    st.error("找不到資料檔案。")
    st.stop()


# ==========================================
# 2. 載入台南市 GeoJSON 地理邊界
# ==========================================
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/ronnywang/twgeojson/master/twtown2010.3.json"
    response = requests.get(url)
    geojson = response.json()
    
    tainan_features = []
    for feature in geojson['features']:
        props = feature.get('properties', {})
        props_str = " ".join(str(v) for v in props.values())
        if '臺南' in props_str or '台南' in props_str:
            town_name = ""
            for key in ['TOWNNAME', 'T_Name', 'name', 'TOWN', 'Town_Name', 'town', 'Town']:
                if key in props:
                    val = str(props[key]).replace('臺南市', '').replace('台南市', '').replace('臺南縣', '').replace('台南縣', '').strip()
                    if len(val) > 0:
                        town_name = val
                        break
            
            if town_name:
                if town_name.endswith('鄉') or town_name.endswith('鎮') or town_name.endswith('市'):
                    town_name = town_name[:-1] + '區'
                feature['id'] = town_name
                tainan_features.append(feature)
                
    geojson['features'] = tainan_features
    return geojson

tainan_geojson = load_geojson()


# ==========================================
# 側邊欄：互動式控制面板
# ==========================================
st.sidebar.title("⚙️ 戰情室控制面板")

# 讓使用者可以獨立選擇左右兩張地圖要看什麼指標
# 等你未來加入預測模型，就可以左邊選「真實 RT_level」，右邊選「預測 RT_level」
metric_options = ['Case_Count', 'RT_level', 'BI', 'CI', 'HI']
left_metric = st.sidebar.selectbox("🗺️ 左側地圖指標", options=metric_options, index=0) # 預設選 Case_Count
right_metric = st.sidebar.selectbox("🗺️ 右側地圖指標", options=metric_options, index=1) # 預設選 RT_level

st.sidebar.markdown("---")
available_years = sorted(df['Year'].unique())
selected_year = st.sidebar.selectbox("📅 選擇年份", available_years)

available_weeks = sorted(df[df['Year'] == selected_year]['Week'].unique())
selected_week = st.sidebar.select_slider(
    "⏱️ 調整週次觀察疫情變化", 
    options=available_weeks,
    value=available_weeks[0]
)


# ==========================================
# 畫圖共用函數 (為了畫左右兩張圖，我們把它包裝成函數)
# ==========================================
def draw_map(df_plot, metric_name):
    color_range = [0, 3] if 'RT_level' in metric_name else None
    
    fig = px.choropleth(
        df_plot,
        geojson=tainan_geojson,
        locations='Town',
        color=metric_name,
        color_continuous_scale="Reds",
        range_color=color_range,
        hover_name='Town',
        # 加入更豐富的懸浮資訊，游標移過去能看到確切數字
        hover_data={'Town': False, metric_name: True} 
    )
    
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    fig.update_traces(marker_line_width=1.5, marker_line_color="white")
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# ==========================================
# 主畫面：KPI 數據卡與雙地圖
# ==========================================
st.title("🦟 台南市登革熱戰情儀表板")

mask = (df['Year'] == selected_year) & (df['Week'] == selected_week)
df_filtered = df[mask]

if df_filtered.empty:
    st.info("這一週沒有任何數據。請嘗試拖動左側的週次滑桿！")
else:
    # 🌟 升級二：戰情室 KPI 數據卡
    st.markdown("### 📊 當週疫情概況")
    
    # 計算 KPI 數值
    total_cases = int(df_filtered['Case_Count'].sum()) if 'Case_Count' in df_filtered.columns else 0
    max_rt = int(df_filtered['RT_level'].max()) if 'RT_level' in df_filtered.columns else 0
    # 假設 RT_level >= 2 算是高風險區域
    high_risk_count = len(df_filtered[df_filtered['RT_level'] >= 2]) if 'RT_level' in df_filtered.columns else 0
    
    # 建立 3 個並排的數據卡
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="🦠 當週總病例數", value=f"{total_cases} 人")
    kpi2.metric(label="🚨 最高警戒等級", value=f"Level {max_rt}")
    kpi3.metric(label="🚩 高風險區塊數量 (Level 2 以上)", value=f"{high_risk_count} 區")
    
    st.markdown("---")
    
    # 🌟 升級一：雙地圖對照
    map_col1, map_col2 = st.columns(2)
    
    with map_col1:
        st.subheader(f"📍 {left_metric}")
        if df_filtered[left_metric].sum() > 0 or 'RT_level' in left_metric:
            fig_left = draw_map(df_filtered, left_metric)
            st.plotly_chart(fig_left, use_container_width=True)
        else:
            st.info("該指標無數據")
            
    with map_col2:
        st.subheader(f"📍 {right_metric}")
        if df_filtered[right_metric].sum() > 0 or 'RT_level' in right_metric:
            fig_right = draw_map(df_filtered, right_metric)
            st.plotly_chart(fig_right, use_container_width=True)
        else:
            st.info("該指標無數據")

st.markdown("---")
with st.expander("點擊查看當週地圖對應的原始資料"):
    st.dataframe(df_filtered)
