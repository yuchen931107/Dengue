import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 將網頁設定為寬螢幕模式 (更適合戰情室與雙地圖)
st.set_page_config(page_title="台南登革熱預測分析", layout="wide", page_icon="🦟")

# ==========================================
# 1. 讀取與清理資料
# ==========================================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data.csv")
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df['Town'] = df['Town'].astype(str).str.replace("台南市", "").str.replace("臺南市", "").str.strip()
    # 建立一個「年-週」的排序鍵，讓跨年份的趨勢圖可以正確排序
    df['YearWeek'] = df['Year'].astype(str) + "-W" + df['Week'].astype(str).str.zfill(2)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ 找不到資料檔案 data.csv，請確認檔案是否與 app.py 放在同一個資料夾。")
    st.stop()


# ==========================================
# 2. 載入台南市 GeoJSON 地理邊界（加上容錯處理）
# ==========================================
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/ronnywang/twgeojson/master/twtown2010.3.json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        geojson = response.json()
    except (requests.RequestException, ValueError) as e:
        return None, str(e)

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
    return geojson, None

tainan_geojson, geojson_error = load_geojson()
if tainan_geojson is None:
    st.warning(f"⚠️ 地理邊界資料載入失敗，地圖功能將無法顯示（錯誤：{geojson_error}）。趨勢分析與排行榜功能不受影響。")


# ==========================================
# 模型設定（之後要加第四個模型，只要改這裡就好）
# ==========================================
MODEL_COLUMNS = ['LSTM', 'SARIMAX', 'XGboost']
MODEL_COLORS = {
    'LSTM': '#1f77b4',
    'SARIMAX': '#2ca02c',
    'XGboost': '#9467bd',
}
available_models = [m for m in MODEL_COLUMNS if m in df.columns]


# ==========================================
# 側邊欄：互動式控制面板
# ==========================================
st.sidebar.title("⚙️ 戰情室控制面板")

metric_options = ['Case_Count', 'RT_level'] + available_models
metric_labels = {
    'Case_Count': '病例數',
    'RT_level': '風險等級 (RT_level)',
    'LSTM': 'LSTM 預測值',
    'SARIMAX': 'SARIMAX 預測值',
    'XGboost': 'XGboost 預測值',
}

left_metric = st.sidebar.selectbox(
    "🗺️ 左側地圖指標", options=metric_options, index=0,
    format_func=lambda x: metric_labels.get(x, x)
)
right_metric = st.sidebar.selectbox(
    "🗺️ 右側地圖指標", options=metric_options, index=1,
    format_func=lambda x: metric_labels.get(x, x)
)

st.sidebar.markdown("---")
available_years = sorted(df['Year'].unique())
selected_year = st.sidebar.selectbox("📅 選擇年份", available_years, index=len(available_years) - 1)

available_weeks = sorted(df[df['Year'] == selected_year]['Week'].unique())
selected_week = st.sidebar.select_slider(
    "⏱️ 調整週次觀察疫情變化",
    options=available_weeks,
    value=available_weeks[-1]
)

st.sidebar.markdown("---")
all_towns = sorted(df['Town'].unique())
selected_town_trend = st.sidebar.selectbox("📍 選擇區域看趨勢（下方頁籤使用）", options=["全市加總"] + all_towns)

st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 趨勢頁籤：要比較哪些模型？**")
selected_models = st.sidebar.multiselect(
    "選擇要疊圖比較的預測模型",
    options=available_models,
    default=available_models,
    format_func=lambda x: metric_labels.get(x, x)
)


# ==========================================
# 畫圖共用函數
# ==========================================
def draw_map(df_plot, metric_name):
    color_range = [0, 3] if metric_name == 'RT_level' else None

    fig = px.choropleth(
        df_plot,
        geojson=tainan_geojson,
        locations='Town',
        color=metric_name,
        color_continuous_scale="Reds",
        range_color=color_range,
        hover_name='Town',
        hover_data={'Town': False, metric_name: True}
    )

    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    fig.update_traces(marker_line_width=1.5, marker_line_color="white")
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def get_previous_week_data(df, year, week):
    """取得上一週的資料，用來計算 KPI 的增減幅度"""
    weeks_in_year = sorted(df[df['Year'] == year]['Week'].unique())
    idx = weeks_in_year.index(week) if week in weeks_in_year else -1
    if idx > 0:
        prev_week = weeks_in_year[idx - 1]
        return df[(df['Year'] == year) & (df['Week'] == prev_week)]
    else:
        # 嘗試往前一年最後一週
        prev_years = [y for y in sorted(df['Year'].unique()) if y < year]
        if prev_years:
            prev_year = prev_years[-1]
            prev_weeks = sorted(df[df['Year'] == prev_year]['Week'].unique())
            if prev_weeks:
                return df[(df['Year'] == prev_year) & (df['Week'] == prev_weeks[-1])]
    return pd.DataFrame()


# ==========================================
# 主畫面
# ==========================================
st.title("🦟 台南市登革熱戰情儀表板")

mask = (df['Year'] == selected_year) & (df['Week'] == selected_week)
df_filtered = df[mask].copy()
df_prev = get_previous_week_data(df, selected_year, selected_week)

tab_map, tab_trend, tab_data = st.tabs(["🗺️ 地圖總覽", "📈 趨勢分析", "📋 原始資料"])

# ------------------------------------------
# 頁籤一：地圖總覽
# ------------------------------------------
with tab_map:
    if df_filtered.empty:
        st.info("這一週沒有任何數據。請嘗試拖動左側的週次滑桿！")
    else:
        st.markdown("### 📊 當週疫情概況")

        total_cases = int(df_filtered['Case_Count'].sum()) if 'Case_Count' in df_filtered.columns else 0
        max_rt = int(df_filtered['RT_level'].max()) if 'RT_level' in df_filtered.columns else 0
        high_risk_count = len(df_filtered[df_filtered['RT_level'] >= 2]) if 'RT_level' in df_filtered.columns else 0

        # 計算與上週的差異
        if not df_prev.empty:
            prev_total_cases = int(df_prev['Case_Count'].sum()) if 'Case_Count' in df_prev.columns else 0
            prev_max_rt = int(df_prev['RT_level'].max()) if 'RT_level' in df_prev.columns else 0
            prev_high_risk = len(df_prev[df_prev['RT_level'] >= 2]) if 'RT_level' in df_prev.columns else 0
            cases_delta = total_cases - prev_total_cases
            rt_delta = max_rt - prev_max_rt
            risk_delta = high_risk_count - prev_high_risk
        else:
            cases_delta = rt_delta = risk_delta = None

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(label="🦠 當週總病例數", value=f"{total_cases} 人",
                    delta=f"{cases_delta:+d} 人" if cases_delta is not None else None,
                    delta_color="inverse")
        kpi2.metric(label="🚨 最高警戒等級", value=f"Level {max_rt}",
                    delta=f"{rt_delta:+d}" if rt_delta is not None else None,
                    delta_color="inverse")
        kpi3.metric(label="🚩 高風險區塊數量 (Level 2 以上)", value=f"{high_risk_count} 區",
                    delta=f"{risk_delta:+d} 區" if risk_delta is not None else None,
                    delta_color="inverse")

        if high_risk_count >= 5:
            st.error(f"⚠️ 目前有 {high_risk_count} 個區域達到高風險等級，建議加強稽查與孳生源清除！")
        elif high_risk_count > 0:
            st.warning(f"提醒：目前有 {high_risk_count} 個區域達到高風險等級。")

        st.markdown("---")

        if tainan_geojson is not None:
            map_col1, map_col2 = st.columns(2)

            with map_col1:
                st.subheader(f"📍 {metric_labels.get(left_metric, left_metric)}")
                if left_metric in df_filtered.columns and (df_filtered[left_metric].sum() > 0 or left_metric == 'RT_level'):
                    fig_left = draw_map(df_filtered, left_metric)
                    st.plotly_chart(fig_left, use_container_width=True)
                else:
                    st.info("該指標無數據")

            with map_col2:
                st.subheader(f"📍 {metric_labels.get(right_metric, right_metric)}")
                if right_metric in df_filtered.columns and (df_filtered[right_metric].sum() > 0 or right_metric == 'RT_level'):
                    fig_right = draw_map(df_filtered, right_metric)
                    st.plotly_chart(fig_right, use_container_width=True)
                else:
                    st.info("該指標無數據")
        else:
            st.info("地理邊界資料無法載入，暫時無法顯示地圖，改看下方排行榜。")

        st.markdown("---")
        st.markdown("### 🏆 當週病例數排行榜（Top 10）")
        if 'Case_Count' in df_filtered.columns:
            top10 = df_filtered.nlargest(10, 'Case_Count').sort_values('Case_Count')
            fig_bar = px.bar(
                top10, x='Case_Count', y='Town', orientation='h',
                color='Case_Count', color_continuous_scale='Reds',
                text='Case_Count'
            )
            fig_bar.update_layout(
                yaxis_title="", xaxis_title="病例數",
                margin={"r": 10, "t": 10, "l": 10, "b": 10},
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------
# 頁籤二：趨勢分析
# ------------------------------------------
with tab_trend:
    st.markdown(f"### 📈 {selected_town_trend} — 歷史趨勢")

    if selected_town_trend == "全市加總":
        agg_dict = {
            'Case_Count': ('Case_Count', 'sum'),
            'RT_level': ('RT_level', 'mean'),
        }
        for m in available_models:
            agg_dict[m] = (m, 'mean')
        trend_df = df.groupby(['Year', 'Week', 'YearWeek'], as_index=False).agg(**agg_dict)
    else:
        trend_df = df[df['Town'] == selected_town_trend].copy()

    trend_df = trend_df.sort_values(['Year', 'Week'])

    # 可以縮小觀察範圍，避免全部年份疊在一起看不清楚
    year_range = st.select_slider(
        "🔎 縮小趨勢圖的觀察年份範圍",
        options=available_years,
        value=(available_years[max(0, len(available_years) - 3)], available_years[-1])
    )
    trend_df_view = trend_df[(trend_df['Year'] >= year_range[0]) & (trend_df['Year'] <= year_range[1])]

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**病例數變化**")
        fig_cases = go.Figure()
        fig_cases.add_trace(go.Scatter(
            x=trend_df_view['YearWeek'], y=trend_df_view['Case_Count'],
            mode='lines+markers', name='實際病例數',
            line=dict(color='#d62728', width=2)
        ))
        fig_cases.update_layout(
            xaxis_title="年-週", yaxis_title="病例數",
            margin={"r": 10, "t": 10, "l": 10, "b": 10},
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_cases, use_container_width=True)

    with col_t2:
        model_label_str = "、".join(metric_labels.get(m, m) for m in selected_models) if selected_models else "（尚未選擇模型）"
        st.markdown(f"**風險等級變化：實際值 vs {model_label_str}**")

        fig_rt = go.Figure()
        fig_rt.add_trace(go.Scatter(
            x=trend_df_view['YearWeek'], y=trend_df_view['RT_level'],
            mode='lines+markers', name='實際 RT_level',
            line=dict(color='#ff7f0e', width=3)
        ))
        for m in selected_models:
            if m in trend_df_view.columns:
                fig_rt.add_trace(go.Scatter(
                    x=trend_df_view['YearWeek'], y=trend_df_view[m],
                    mode='lines+markers', name=f'{m} 預測',
                    line=dict(color=MODEL_COLORS.get(m, '#888888'), width=2, dash='dash')
                ))
        fig_rt.update_layout(
            xaxis_title="年-週", yaxis_title="RT_level",
            margin={"r": 10, "t": 10, "l": 10, "b": 10},
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        fig_rt.add_hline(y=2, line_dash="dot", line_color="red",
                          annotation_text="高風險門檻", annotation_position="top left")
        st.plotly_chart(fig_rt, use_container_width=True)

    # 模型準確度比較表：每個模型各自算一次 MAE 與等級命中率，方便一次比較三個模型
    if selected_models:
        st.markdown("#### 📐 模型誤差比較")
        rows = []
        for m in selected_models:
            if m in trend_df_view.columns:
                valid = trend_df_view[['RT_level', m]].dropna()
                if not valid.empty:
                    mae = (valid['RT_level'] - valid[m]).abs().mean()
                    acc = (valid['RT_level'].round() == valid[m].round()).mean() * 100
                    rows.append({"模型": m, "MAE（級）": round(mae, 3), "等級命中率": f"{acc:.1f}%"})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("模型"), use_container_width=True)
    else:
        st.info("請從左側選單勾選至少一個模型才會顯示比較圖與誤差表。")

# ------------------------------------------
# 頁籤三：原始資料
# ------------------------------------------
with tab_data:
    st.markdown(f"### 📋 {selected_year} 年第 {selected_week} 週 原始資料")
    st.dataframe(df_filtered, use_container_width=True)

    csv_data = df_filtered.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 下載當週資料 (CSV)",
        data=csv_data,
        file_name=f"tainan_dengue_{selected_year}_W{selected_week}.csv",
        mime="text/csv"
    )
