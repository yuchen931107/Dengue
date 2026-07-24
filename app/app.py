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
    df = pd.read_csv(file_path, encoding='big5')
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

# RT_level 是類別型的風險等級（0～3），不是連續數值，所以地圖與趨勢圖都用離散配色，
# 而不是用連續色階去內插出中間的顏色。
RT_LEVEL_COLORS = {
    '0': '#2ca02c',  # 綠：無風險
    '1': '#ffd166',  # 黃：低度關注
    '2': '#ff7f0e',  # 橘：高風險
    '3': '#d62728',  # 紅：最高風險
}
RT_LEVEL_ORDER = ['0', '1', '2', '3']

metric_options = ['Case_Count', 'RT_level'] + available_models
metric_labels = {
    'Case_Count': '病例數',
    'RT_level': '風險等級 (RT_level)',
    'LSTM': 'LSTM 預測值',
    'SARIMAX': 'SARIMAX 預測值',
    'XGboost': 'XGboost 預測值',
}

available_years = sorted(df['Year'].unique())
all_towns = sorted(df['Town'].unique())


# ==========================================
# 共用函數（年份/週次篩選、地圖繪製都集中在這裡，避免各分頁各寫一份）
# ==========================================
def weeks_in_year(year):
    """回傳某一年份中，資料裡實際存在的所有週次（由小到大排序）"""
    return sorted(df[df['Year'] == year]['Week'].unique())


def filter_by_week(year, week):
    """依年份+週次篩出當週資料"""
    return df[(df['Year'] == year) & (df['Week'] == week)].copy()



def draw_map(df_plot, metric_name):
    if metric_name == 'RT_level':
        # 類別型資料：先四捨五入成整數等級，再轉成字串類別，才能用離散配色
        # （而不是像 Case_Count 那樣用連續色階去內插）。
        df_plot = df_plot.copy()
        df_plot['RT_level_cat'] = df_plot['RT_level'].round().astype('Int64').astype(str)

        fig = px.choropleth(
            df_plot,
            geojson=tainan_geojson,
            locations='Town',
            color='RT_level_cat',
            color_discrete_map=RT_LEVEL_COLORS,
            category_orders={'RT_level_cat': RT_LEVEL_ORDER},
            hover_name='Town',
            hover_data={'Town': False, 'RT_level_cat': False, 'RT_level': True}
        )
        fig.update_layout(legend_title_text="風險等級")
    else:
        fig = px.choropleth(
            df_plot,
            geojson=tainan_geojson,
            locations='Town',
            color=metric_name,
            color_continuous_scale="Reds",
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


def get_previous_week_data(year, week):
    """取得上一週的資料，用來計算 KPI 的增減幅度"""
    weeks = weeks_in_year(year)
    idx = weeks.index(week) if week in weeks else -1
    if idx > 0:
        return filter_by_week(year, weeks[idx - 1])

    # 當週是該年第一週時，嘗試往前一年最後一週找
    prev_years = [y for y in available_years if y < year]
    if prev_years:
        prev_year = prev_years[-1]
        prev_weeks = weeks_in_year(prev_year)
        if prev_weeks:
            return filter_by_week(prev_year, prev_weeks[-1])
    return pd.DataFrame()


def render_map_column(column, metric_name, df_filtered):
    """畫出單一欄位的地圖：先檢查該指標有沒有數據，再呼叫 draw_map"""
    with column:
        st.subheader(f"📍 {metric_labels.get(metric_name, metric_name)}")
        has_data = metric_name in df_filtered.columns and (
            df_filtered[metric_name].sum() > 0 or metric_name == 'RT_level'
        )
        if has_data:
            st.plotly_chart(draw_map(df_filtered, metric_name), use_container_width=True)
        else:
            st.info("該指標無數據")


# ==========================================
# 主畫面
# ==========================================
st.title("🦟 台南市登革熱戰情儀表板")

tab_map, tab_trend, tab_data = st.tabs(["🗺️ 地圖總覽", "📈 趨勢分析", "📋 原始資料"])

# ------------------------------------------
# 頁籤一：地圖總覽（控制選項在頁籤內）
# ------------------------------------------
with tab_map:
    st.markdown("#### ⚙️ 本頁控制選項")
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1.4])

    with ctrl1:
        left_metric = st.selectbox(
            "🗺️ 左側地圖指標", options=metric_options, index=0,
            format_func=lambda x: metric_labels.get(x, x), key="map_left_metric"
        )
    with ctrl2:
        right_metric = st.selectbox(
            "🗺️ 右側地圖指標", options=metric_options, index=1,
            format_func=lambda x: metric_labels.get(x, x), key="map_right_metric"
        )
    with ctrl3:
        selected_year = st.selectbox(
            "📅 選擇年份", available_years, index=len(available_years) - 1, key="map_year"
        )

    available_weeks_map = weeks_in_year(selected_year)
    with ctrl4:
        selected_week = st.select_slider(
            "⏱️ 調整週次觀察疫情變化",
            options=available_weeks_map,
            value=available_weeks_map[-1],
            key="map_week"
        )

    st.markdown("---")

    df_filtered = filter_by_week(selected_year, selected_week)
    df_prev = get_previous_week_data(selected_year, selected_week)

    if df_filtered.empty:
        st.info("這一週沒有任何數據。請嘗試拖動上方的週次滑桿！")
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
            render_map_column(map_col1, left_metric, df_filtered)
            render_map_column(map_col2, right_metric, df_filtered)
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
# 頁籤二：趨勢分析（控制選項在頁籤內）
# ------------------------------------------
with tab_trend:
    st.markdown("#### ⚙️ 本頁控制選項")
    tctrl1, tctrl2 = st.columns([1, 2])

    with tctrl1:
        selected_town_trend = st.selectbox(
            "📍 選擇區域看趨勢", options=["全市加總"] + all_towns, key="trend_town"
        )
    with tctrl2:
        selected_models = st.multiselect(
            "🤖 選擇要疊圖比較的預測模型",
            options=available_models,
            default=available_models,
            format_func=lambda x: metric_labels.get(x, x),
            key="trend_models"
        )

    year_range = st.select_slider(
        "🔎 縮小趨勢圖的觀察年份範圍",
        options=available_years,
        value=(available_years[max(0, len(available_years) - 3)], available_years[-1]),
        key="trend_year_range"
    )

    st.markdown("---")
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
        # RT_level 是類別型的等級（0/1/2/3），不是連續數值，所以實際值用階梯線
        # （line_shape='hv'）呈現「跳到下一級」的感覺，而不是用直線內插出 1.5 級這種不存在的值。
        fig_rt.add_trace(go.Scatter(
            x=trend_df_view['YearWeek'], y=trend_df_view['RT_level'],
            mode='lines+markers', name='實際 RT_level',
            line=dict(color='#ff7f0e', width=3, shape='hv')
        ))
        for m in selected_models:
            if m in trend_df_view.columns:
                fig_rt.add_trace(go.Scatter(
                    x=trend_df_view['YearWeek'], y=trend_df_view[m],
                    mode='lines+markers', name=f'{m} 預測',
                    line=dict(color=MODEL_COLORS.get(m, '#888888'), width=2, dash='dash', shape='hv')
                ))
        fig_rt.update_layout(
            xaxis_title="年-週", yaxis_title="RT_level（等級）",
            yaxis=dict(tickmode='array', tickvals=[0, 1, 2, 3], range=[-0.3, 3.3]),
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

        st.markdown("#### 🔢 混淆矩陣：實際等級 vs 預測等級")
        st.caption("對角線（左上到右下）代表猜對；離對角線越遠，代表猜錯的級數差越大。")

        cm_models = [m for m in selected_models if m in trend_df_view.columns]
        if cm_models:
            cm_cols = st.columns(len(cm_models))
            for col, m in zip(cm_cols, cm_models):
                valid = trend_df_view[['RT_level', m]].dropna()
                with col:
                    st.markdown(f"**{m}**")
                    if valid.empty:
                        st.info("沒有可比對的資料")
                        continue

                    # 四捨五入並限制在 0~3 級之間，避免模型輸出超出範圍的極端值
                    actual_cat = valid['RT_level'].round().clip(0, 3).astype(int)
                    pred_cat = valid[m].round().clip(0, 3).astype(int)

                    cm = pd.crosstab(actual_cat, pred_cat)
                    cm = cm.reindex(index=[0, 1, 2, 3], columns=[0, 1, 2, 3], fill_value=0)

                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm.values,
                        x=[f"預測 {c}" for c in cm.columns],
                        y=[f"實際 {r}" for r in cm.index],
                        colorscale='Blues',
                        text=cm.values,
                        texttemplate="%{text}",
                        showscale=False,
                        hovertemplate="實際等級 %{y}<br>預測等級 %{x}<br>次數：%{z}<extra></extra>"
                    ))
                    fig_cm.update_layout(
                        margin={"r": 10, "t": 10, "l": 10, "b": 10},
                        yaxis=dict(autorange='reversed')
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("請從上方選單勾選至少一個模型才會顯示比較圖與誤差表。")

# ------------------------------------------
# 頁籤三：原始資料（控制選項在頁籤內）
# ------------------------------------------
with tab_data:
    st.markdown("#### ⚙️ 本頁控制選項")
    dctrl1, dctrl2 = st.columns(2)

    with dctrl1:
        data_year = st.selectbox("📅 選擇年份", available_years, index=len(available_years) - 1, key="data_year")

    available_weeks_data = weeks_in_year(data_year)
    with dctrl2:
        data_week = st.select_slider(
            "⏱️ 選擇週次", options=available_weeks_data, value=available_weeks_data[-1], key="data_week"
        )

    st.markdown("---")

    df_filtered_data = filter_by_week(data_year, data_week)

    st.markdown(f"### 📋 {data_year} 年第 {data_week} 週 原始資料")
    st.dataframe(df_filtered_data, use_container_width=True)

    csv_data = df_filtered_data.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 下載當週資料 (CSV)",
        data=csv_data,
        file_name=f"tainan_dengue_{data_year}_W{data_week}.csv",
        mime="text/csv"
    )
