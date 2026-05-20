import pandas as pd
import numpy as np
import itertools

# ==========================================
# 1. 讀取資料
# ==========================================
cases = pd.read_csv("Tainan_cases_data.csv")
weather = pd.read_csv("Tainan_History_Weather_2010_2026.csv")
invest = pd.read_csv("Tainan_invest_data.csv")
pop = pd.read_csv("tainan_population_final.csv")

# ==========================================
#2.建立連續的「時空網格」底表 (Base Grid)
# ==========================================
towns = pop['區域別'].dropna().unique()
towns = [t for t in towns if t != '總計']

#保持從 2010 年開始建網格，為了讓後面 2011 年初的 Lag 特徵有前置資料可抓
date_range = pd.date_range(start='2010-01-01', end='2025-12-31', freq='W-MON')

grid = list(itertools.product(date_range, towns))
base_df = pd.DataFrame(grid, columns=['Week', 'Town'])

base_df['Year'] = base_df['Week'].dt.year
base_df['Month'] = base_df['Week'].dt.month

# ==========================================
#3.處理「目標變數 Y」: 確診病例數
# ==========================================
cases['發病日'] = pd.to_datetime(cases['發病日'])
cases = cases.dropna(subset=['發病日', '居住鄉鎮'])
cases['Week'] = cases['發病日'].dt.to_period('W').dt.start_time

cases_agg = cases.groupby(['Week', '居住鄉鎮']).size().reset_index(name='Case_Count')

model_df = pd.merge(base_df, cases_agg, left_on=['Week', 'Town'], right_on=['Week', '居住鄉鎮'], how='left')
model_df['Case_Count'] = model_df['Case_Count'].fillna(0)
model_df = model_df.drop(columns=['居住鄉鎮'])

# ==========================================
#4.處理「特徵 X」: 天氣資料 (包含 Lag)
# ==========================================
weather['日期'] = pd.to_datetime(weather['日期'])
weather['Week'] = weather['日期'].dt.to_period('W').dt.start_time

weather_agg = weather.groupby('Week').agg({
    '平均氣溫(℃)': 'mean',
    '日累積降水量(mm)': 'sum'
}).reset_index()

weather_agg['Temp_Lag2W'] = weather_agg['平均氣溫(℃)'].shift(2)
weather_agg['Rain_Lag4W'] = weather_agg['日累積降水量(mm)'].shift(4)

model_df = pd.merge(model_df, weather_agg, on='Week', how='left')

# ==========================================
#5.處理「特徵 X」: 病媒蚊指數 (包含 Lag 與 Ffill)
# ==========================================
invest['Date'] = pd.to_datetime(invest['Date'])
invest['Week'] = invest['Date'].dt.to_period('W').dt.start_time

invest_agg = invest.groupby(['Week', 'Town'])['BI'].mean().reset_index(name='Avg_BI')
invest_agg = invest_agg.sort_values(['Town', 'Week'])
invest_agg['Avg_BI_Lag2W'] = invest_agg.groupby('Town')['Avg_BI'].shift(2)

model_df = pd.merge(model_df, invest_agg, on=['Week', 'Town'], how='left')

#利用 2010 年的資料往下填補 2011 年初可能缺失的調查紀錄
model_df['Avg_BI'] = model_df.groupby('Town')['Avg_BI'].ffill().fillna(0)
model_df['Avg_BI_Lag2W'] = model_df.groupby('Town')['Avg_BI_Lag2W'].ffill().fillna(0)

# ==========================================
# 6.處理「特徵 X」: 人口密度
# ==========================================
pop[['ROC_Year', 'Month']] = pop['年月'].astype(str).str.split('.', expand=True).astype(int)
pop['Year'] = pop['ROC_Year'] + 1911
pop_clean = pop[['Year', 'Month', '區域別', '人口密度']].rename(columns={'區域別': 'Town'})

pop_clean['人口密度'] = pd.to_numeric(pop_clean['人口密度'], errors='coerce')

# 如果原本資料是各里的密度，這裡取平均會得出該區的概略密度；如果是重複的總計行，取平均依然是總計。
pop_clean = pop_clean.groupby(['Year', 'Month', 'Town'])['人口密度'].mean().reset_index()

model_df = pd.merge(model_df, pop_clean, on=['Year', 'Month', 'Town'], how='left')

model_df['人口密度'] = model_df.groupby('Town')['人口密度'].ffill().fillna(0)

# ==========================================
#7.最終篩選：只保留 2011 年(含)以後的資料
# ==========================================
model_df = model_df[model_df['Week'] >= '2011-01-01']

# 清理欄位與重新排序
model_df = model_df.sort_values(['Town', 'Week']).reset_index(drop=True)
model_df.rename(columns={'平均氣溫(℃)': 'Avg_Temp', '日累積降水量(mm)': 'rain(mm)','人口密度':'Population density'}, inplace=True)

model_df.to_csv("Tainan_Dengue_ML.csv", index=False, encoding='utf-8-sig')
