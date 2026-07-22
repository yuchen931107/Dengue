import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="台南登革熱熱力圖", layout="wide")

# 1. 讀取資料
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("找不到 data.csv 檔案，請確認檔案位置。")
    st.stop()

# ==========================================
# 資料預處理：自動配對台南市各區經緯度
# ==========================================
# 建立台南各區中心點座標的對照表 (緯度 lat, 經度 lon)
tainan_coords = {
    "中西區": [22.991, 120.203], "東區": [22.980, 120.222], "南區": [22.961, 120.187],
    "北區": [23.004, 120.203], "安平區": [22.993, 120.165], "安南區": [23.048, 120.185],
    "永康區": [23.026, 120.260], "歸仁區": [22.967, 120.293], "新化區": [23.038, 120.310],
    "左鎮區": [23.042, 120.408], "玉井區": [23.123, 120.463], "楠西區": [23.174, 120.485],
    "南化區": [23.051, 120.477], "仁德區": [22.973, 120.252], "關廟區": [22.963, 120.328],
    "龍崎區": [22.965, 120.362], "官田區": [23.194, 120.315], "麻豆區": [23.185, 120.250],
    "佳里區": [23.166, 120.177], "西港區": [23.123, 120.203], "七股區": [23.141, 120.140],
    "將軍區": [23.199, 120.157], "學甲區": [23.232, 120.181], "北門區": [23.267, 120.125],
    "新營區": [23.310, 120.316], "後壁區": [23.366, 120.362], "白河區": [23.344, 120.415],
    "東山區": [23.326, 120.404], "六甲區": [23.231, 120.347], "下營區": [23.235, 120.264],
    "柳營區": [23.279, 120.316], "鹽水區": [23.319, 120.266], "善化區": [23.132, 120.295],
    "大內區": [23.118, 120.350], "山上區": [23.104, 120.352], "新市區": [23.078, 120.294],
    "安定區": [23.121, 120.236]
}

# 智慧清理欄位：預防資料集裡面的文字包含「台南市」或「臺南市」
def get_coords(town_name):
    clean_name = str(town_name).replace("台南市", "").replace("臺南市", "").strip()
    return tainan_coords.get(clean_name, [None, None])

# 將座標加入資料集中
df['lat'] = df['Town'].apply(lambda x: get_coords(x)[0])
df['lon'] = df['Town'].apply(lambda x: get_coords(x)[1])

# 過濾掉找不到座標的資料 (確保地圖不會報錯)
df_map = df.dropna(subset=['lat', 'lon'])


# ==========================================
# 側邊欄：互動式控制面板
# ==========================================
st.sidebar.title("⚙️ 熱力圖控制面板")

# 選擇要觀察的權重 (熱力點的大小與顏色深淺)
heat_metric = st.sidebar.selectbox(
    "選擇熱力圖指標", 
    options=['Case_Count', 'RT_level', 'BI', 'CI', 'HI']
)

# 選擇年份
available_years = sorted(df_map['Year'].unique())
selected_year = st.sidebar.selectbox("選擇年份", available_years)

# 選擇該年份的週次 (改用 select_slider 完美支援日期格式)
# 1. 抓出該年份所有出現過的日期，並確保照時間排序
available_weeks = sorted(df_map[df_map['Year'] == selected_year]['Week'].unique())

# 2. 使用 select_slider 讓使用者在這些日期中滑動
selected_week = st.sidebar.select_slider(
    "調整週次(日期)觀察疫情擴散", 
    options=available_weeks,
    value=available_weeks[0]  # 預設停在第一週
)


# ==========================================
# 主畫面：熱力圖渲染
# ==========================================
st.title("🗺️ 台南市登革熱疫情熱力圖")
st.write(f"目前顯示：**{selected_year} 年 第 {selected_week} 週**的 `{heat_metric}` 分布狀況")

# 根據使用者的選擇過濾資料
mask = (df_map['Year'] == selected_year) & (df_map['Week'] == selected_week)
df_filtered = df_map[mask]

if df_filtered.empty or df_filtered[heat_metric].sum() == 0:
    st.info("這一週沒有任何數據或數值皆為零，熱力圖無法成型。請嘗試拖動左側的週次滑桿！")
else:
    # 使用 Plotly Express 繪製地圖
    fig = px.density_mapbox(
        df_filtered, 
        lat='lat', 
        lon='lon', 
        z=heat_metric,        # 決定熱力圖深淺的欄位
        radius=25,            # 熱力點的擴散半徑
        center=dict(lat=23.1, lon=120.25), # 預設中心點對準台南
        zoom=9,               # 預設縮放大小
        mapbox_style="carto-positron",     # 乾淨的底圖風格
        hover_name="Town",    # 滑鼠移過去顯示區域名稱
        color_continuous_scale="Reds"      # 熱力圖顏色 (以紅色為主)
    )
    
    # 讓地圖邊距歸零，看起來更滿版
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    # 顯示在地圖上
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
with st.expander("點擊查看當週地圖對應的原始資料"):
    st.dataframe(df_filtered[['Town', 'Year', 'Week', heat_metric]])