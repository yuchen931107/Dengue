import pandas as pd
import itertools
#If 有缺失值會以前一週來補
# ==========================================
# 1. 讀取資料
# ==========================================
cases = pd.read_csv("Tainan_cases_data.csv")
weather = pd.read_csv("Tainan_History_Weather_2010_2026.csv")
invest = pd.read_csv("Tainan_invest_data.csv")
pop = pd.read_csv("tainan_population_final.csv")
rt = pd.read_csv("Tainan_RT.csv")
rt_subset = rt[['Week','Town','Mean(R)']].copy()
end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=7)
# ==========================================
#2.建立連續的「時空網格」底表 (Base Grid)
# ==========================================
towns = pop['區域別'].dropna().unique()
towns = [t for t in towns if t != '總計']
#設定時間
date_range = pd.date_range(start='2010-1-1', end=end_date, freq='W-MON')
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
model_df = model_df.sort_values(['Town', 'Week'])
model_df['Case_Lag1W'] = model_df.groupby('Town')['Case_Count'].shift(1)
model_df['Case_Lag2W'] = model_df.groupby('Town')['Case_Count'].shift(2)
model_df['Case_Lag3W'] = model_df.groupby('Town')['Case_Count'].shift(3)
# ==========================================
#4.處理「特徵 X」: 天氣資料 (包含 Lag)
# ==========================================
weather['日期'] = pd.to_datetime(weather['日期'])
weather['Week'] = weather['日期'].dt.to_period('W').dt.start_time

weather_agg = weather.groupby('Week').agg({
    '平均氣溫(℃)': 'mean',
    '日累積降水量(mm)': 'sum'
}).reset_index()
for i in range(2,9,2):
    weather_agg[f'Temp_Lag{i}W'] = weather_agg['平均氣溫(℃)'].shift(i)
    weather_agg[f'Rain_Lag{i}W'] = weather_agg['日累積降水量(mm)'].shift(i)
model_df = pd.merge(model_df, weather_agg, on='Week', how='left')
# ==========================================
#5.處理「特徵 X」: 病媒蚊指數 (包含 Lag 與 Ffill)
# ==========================================
metrics = ['BI', 'CI', 'HI', 'LI', 'AI', 'PI', 'Con100HH']
invest['Date'] = pd.to_datetime(invest['Date'])
invest['Week'] = invest['Date'].dt.to_period('W').dt.start_time
invest_agg = invest.groupby(['Week', 'Town'])[metrics].mean().reset_index()
invest_agg = invest_agg.sort_values(['Town', 'Week'])
for col in metrics:
    invest_agg[f'{col}_Lag2W'] = invest_agg.groupby('Town')[f'{col}'].shift(2)
model_df = pd.merge(model_df, invest_agg, on=['Week', 'Town'], how='left')
for col in metrics:
    # 填補當週平均
    model_df[f'{col}'] = model_df.groupby('Town')[f'{col}'].ffill().fillna(0)
    # 填補前兩週平均
    model_df[f'{col}_Lag2W'] = model_df.groupby('Town')[f'{col}_Lag2W'].ffill().fillna(0)
model_df.info()
#北門、南化、善化、學甲、官田、將軍、山上、左鎮、玉井少很多資料
# ==========================================
# 6.處理「特徵 X」: 人口密度
# ==========================================
pop[['ROC_Year', 'Month']] = pop['年月'].astype(str).str.split('.', expand=True).astype(int)
pop['Year'] = pop['ROC_Year'] + 1911
pop_clean = pop[['Year', 'Month', '區域別', '人口密度']].rename(columns={'區域別': 'Town'})
pop_clean['人口密度'] = pd.to_numeric(pop_clean['人口密度'], errors='coerce')
pop_clean = pop_clean.groupby(['Year', 'Month', 'Town'])['人口密度'].mean().reset_index()

model_df = pd.merge(model_df, pop_clean, on=['Year', 'Month', 'Town'], how='left')
model_df['人口密度'] = model_df.groupby('Town')['人口密度'].ffill()
model_df['人口密度'] = model_df.groupby('Town')['人口密度'].bfill()
model_df['人口密度'] = model_df['人口密度'].fillna(0)
# ==========================================
# 7.計算RT
# ==========================================
#model_df['Week'] = pd.to_datetime(model_df['Week'])
rt_subset['Week'] = pd.to_datetime(rt_subset['Week'])
model_df = pd.merge(model_df,rt_subset,on=['Week', 'Town'],how='left')
model_df.rename(columns={'Mean(R)': 'RT'}, inplace=True)
# ==========================================
#8.最終篩選
# ==========================================
model_newest=model_df[model_df['Week'] >= '2011-01-01']
model_df = model_df[model_df['Week'].between('2011-01-01', '2025-12-31')]
# 清理欄位與重新排序
model_df = model_df.sort_values(['Town', 'Week']).reset_index(drop=True)
model_df.rename(columns={'平均氣溫(℃)': 'Avg_Temp',
                         '日累積降水量(mm)':'rain(mm)',
                         '人口密度':'Population density',
                         'Avg_BI':'BI','Avg_BI_Lag2W':'BI_lag2W'}, inplace=True)
model_df.to_csv("Tainan_Dengue_ML.csv", index=False, encoding='utf-8-sig')
model_newest.to_csv("Tainan_Dengue_ML_newest.csv", index=False, encoding='utf-8-sig')
