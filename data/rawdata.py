import pandas as pd
import itertools
import warnings
from RT_split import RT_split
warnings.filterwarnings('ignore')

# ==========================================
# 1. 讀取所有資料集
# ==========================================
print("正在讀取資料...")
cases = pd.read_csv("Tainan_cases_data.csv", low_memory=False)
weather = pd.read_csv("Tainan_History_Weather_2010_2026.csv")
invest = pd.read_csv("Tainan_invest_data.csv", low_memory=False)
rt = pd.read_csv("Tainan_RT.csv")
df_spray = pd.read_csv("medicine_features.csv") 

# ==========================================
# 2. 建立連續的時空底表 (Base Grid)
# ==========================================
print("建立時空底表...")
# 從 RT 資料中抓取台南所有的 Town 作為底表 (排除總計)
towns = rt['Town'].dropna().unique()
towns = [t for t in towns if t != '總計']

# 設定時間 (統一對齊週一)
end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=7)
date_range = pd.date_range(start='2010-01-01', end=end_date, freq='W-MON')

grid = list(itertools.product(date_range, towns))
model_df = pd.DataFrame(grid, columns=['Week', 'Town'])
model_df['Year'] = model_df['Week'].dt.year
model_df['Month'] = model_df['Week'].dt.month

# ==========================================
# 3. 處理「目標變數 Y」: 確診病例數
# ==========================================
print("處理病例資料...")
cases['發病日'] = pd.to_datetime(cases['發病日'])
cases = cases.dropna(subset=['發病日', '居住鄉鎮'])
cases['Week'] = cases['發病日'].dt.to_period('W').dt.start_time

cases_agg = cases.groupby(['Week', '居住鄉鎮']).size().reset_index(name='Case_Count')
model_df = pd.merge(model_df, cases_agg, left_on=['Week', 'Town'], right_on=['Week', '居住鄉鎮'], how='left')
model_df = model_df.drop(columns=['居住鄉鎮'])

# ★ 遺失值先不處理，保留 NaN (不在這邊補 0)

# ==========================================
# 4. 處理「特徵 X」: 天氣資料 
# ==========================================
print("處理氣候資料...")
weather['日期'] = pd.to_datetime(weather['日期'])
weather['Week'] = weather['日期'].dt.to_period('W').dt.start_time

weather_agg = weather.groupby('Week').agg({
    '平均氣溫(℃)': 'mean',
    '溫差(℃)': 'mean',
    '平均相對溼度(%)': 'mean',
    '日照時數(小時)': 'sum',
    '降水時數(小時)': 'sum',
    '日累積降水量(mm)': 'sum'
}).reset_index().rename(columns={
    '平均氣溫(℃)': 'AvgTemp', '溫差(℃)': 'TempRange',
    '平均相對溼度(%)': 'AvgHumidity', '日照時數(小時)': 'SunshineHours',
    '降水時數(小時)': 'RainfallHours', '日累積降水量(mm)': 'Rainfall'
})

model_df = pd.merge(model_df, weather_agg, on='Week', how='left')

# ★ 天氣異常負值與遺失值先不處理，保留原始狀態

# ==========================================
# 5. 處理「特徵 X」: 病媒蚊指數
# ==========================================
print("處理蟲媒調查資料...")
metrics = ['BI', 'CI', 'HI', 'LI', 'AI', 'PI', 'Con100HH']
invest['Date'] = pd.to_datetime(invest['Date'])
invest['Week'] = invest['Date'].dt.to_period('W').dt.start_time

# 強制將無法轉換數值的空值轉為 NaN
invest[metrics] = invest[metrics].apply(pd.to_numeric, errors='coerce')
invest_agg = invest.groupby(['Week', 'Town'])[metrics].mean().reset_index()

model_df = pd.merge(model_df, invest_agg, on=['Week', 'Town'], how='left')

# ★ 蟲媒遺失值先不處理，保留 NaN (不在這邊補 0)

# ==========================================
# 6. 處理「傳染數 RT 資料」
# ==========================================
print("處理 RT 資料...")
rt_subset = rt[['Week', 'Town', 'Mean(R)']].copy()
rt_subset['Week'] = pd.to_datetime(rt_subset['Week'])

model_df = pd.merge(model_df, rt_subset, on=['Week', 'Town'], how='left')
model_df.rename(columns={'Mean(R)': 'RT'}, inplace=True)

model_df = RT_split(model_df, rt_column='RT', new_column='RT_level')

# ==========================================
# 7. 處理「特徵 X」: 噴藥紀錄
# ==========================================
print("處理噴藥資料...")
df_spray['Week'] = pd.to_datetime(df_spray['Week'])
model_df = pd.merge(model_df, df_spray, on=["Week", "Town"], how="left")

# 有噴藥紀錄 (非空值) 轉為 1，沒有紀錄 (NaN) 轉為 0
model_df["Medicine"] = model_df["次數"].notnull().astype(int)

# 移除原始的「次數」欄位
model_df = model_df.drop(columns=["次數"])

# ==========================================
# 8. 排序與輸出
# ==========================================
print("輸出資料...")
# 清理欄位與重新排序
model_df = model_df.sort_values(['Town', 'Week']).reset_index(drop=True)

# 調整時間範圍
model_df = model_df[model_df["Week"] <= "2026-01-01"]
model_df = model_df[model_df["Week"] >= "2011-01-01"]

# 將天氣與 Medicine 欄位移到最後面，保持版面整潔
weather_cols_base = ['AvgTemp', 'TempRange', 'AvgHumidity', 'SunshineHours', 'RainfallHours', 'Rainfall']
other_cols = [c for c in model_df.columns if c not in weather_cols_base and c != 'Medicine']
model_df = model_df[other_cols + weather_cols_base + ['Medicine']]

# 存成 RawData
model_df.to_csv("Tainan_Dengue_RawData.csv", index=False, encoding='utf-8-sig')

print("合併完成！資料已儲存為 'Tainan_Dengue_RawData.csv'")
print(f"rawdata 資料筆數: {model_df.shape[0]}, 欄位數: {model_df.shape[1]}")