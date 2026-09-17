import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 讀取所有資料集
# ==========================================
print("正在讀取資料...")
df_cases = pd.read_csv('Tainan_cases_data.csv', low_memory=False)
df_weather = pd.read_csv('Tainan_History_Weather_2010_2026.csv')
df_invest = pd.read_csv('Tainan_invest_data.csv', low_memory=False)
df_pop = pd.read_csv('tainan_population_final.csv')
df_rt = pd.read_csv('Tainan_RT.csv')

# ==========================================
# 2. 定義時間轉換函數 (統一對齊到當週的星期一)
# ==========================================
def to_week_start(date_series):
    # 轉為 datetime 後，減去 weekday 天數，對齊星期一
    dt = pd.to_datetime(date_series, errors='coerce')
    return dt - pd.to_timedelta(dt.dt.weekday, unit='D')

# ==========================================
# 3. 處理「病例資料」(df_cases) -> 轉每週/每區的 Case_Count
# ==========================================
print("處理病例資料...")
# 過濾台南市，並以發病日為主 (無發病日則用通報日)
df_cases['Date'] = df_cases['發病日'].fillna(df_cases['通報日'])
df_cases = df_cases[(df_cases['居住縣市'] == '台南市') & (df_cases['Date'].notnull())]
df_cases['Week'] = to_week_start(df_cases['Date'])

# 計算每週每區的總病例數
cases_agg = df_cases.groupby(['Week', '居住鄉鎮']).size().reset_index(name='Case_Count')
cases_agg.rename(columns={'居住鄉鎮': 'Town'}, inplace=True)

# ==========================================
# 4. 處理「氣候資料」(df_weather) -> 轉每週平均
# ==========================================
print("處理氣候資料...")
df_weather['Week'] = to_week_start(df_weather['日期'])
# 將非數值欄位排除後取平均 (氣候為全市共用，不分區)
weather_agg = df_weather.drop(columns=['日期']).groupby('Week').mean().reset_index()
weather_agg.rename(columns={
    '平均氣溫(℃)': 'AvgTemp', '溫差(℃)': 'TempRange', 
    '平均相對溼度(%)': 'AvgHumidity', '日照時數(小時)': 'SunshineHours',
    '降水時數(小時)': 'RainfallHours', '日累積降水量(mm)': 'Rainfall'
}, inplace=True)

# ==========================================
# 5. 處理「蟲媒調查資料」(df_invest) -> 轉每週/每區平均
# ==========================================
print("處理蟲媒調查資料...")
df_invest['Week'] = to_week_start(df_invest['Date'])
# 選取重要的蚊子指標取平均
invest_cols = ['BI', 'CI', 'HI', 'LI', 'AI', 'Con100HH']
# 將無法轉換數值的空值填 NaN
df_invest[invest_cols] = df_invest[invest_cols].apply(pd.to_numeric, errors='coerce')
invest_agg = df_invest.groupby(['Week', 'Town'])[invest_cols].mean().reset_index()

# ==========================================
# 6. 處理「人口密度資料」(df_pop) -> 轉為西元年/月，以便後續對接
# ==========================================
print("處理人口密度資料...")
# 年月格式處理 (如 115.4 -> 115年4月)
def parse_tw_year_month(ym):
    ym_str = str(ym)
    if '.' in ym_str:
        y, m = ym_str.split('.')
        return int(y) + 1911, int(m)
    return np.nan, np.nan

df_pop['Year'], df_pop['Month'] = zip(*df_pop['年月'].apply(parse_tw_year_month))
pop_agg = df_pop[['Year', 'Month', '區域別', '人口密度']].rename(columns={'區域別': 'Town', '人口密度': 'Population density'})

# ==========================================
# 7. 處理「傳染數 RT 資料」(df_rt) -> 擷取 Mean(R)
# ==========================================
print("處理 RT 資料...")
df_rt['Week'] = to_week_start(df_rt['Week'])
rt_agg = df_rt[['Week', 'Town', 'Mean(R)']].rename(columns={'Mean(R)': 'RT'})

# ==========================================
# 8. 建立時空骨架 (Base Grid) 並進行大合併 (Merge)
# ==========================================
print("進行資料大合併 (Merge)...")
# 為了避免沒有病例的週次被漏掉，建立一個包含所有 (週次 x 台南市各區) 的基礎表格
all_weeks = pd.date_range(start='2010-01-04', end='2026-05-25', freq='W-MON')
all_towns = df_invest['Town'].dropna().unique()
base_grid = pd.MultiIndex.from_product([all_weeks, all_towns], names=['Week', 'Town']).to_frame(index=False)

# 依序 Merge 所有資料
rawdata = pd.merge(base_grid, cases_agg, on=['Week', 'Town'], how='left')
rawdata['Case_Count'] = rawdata['Case_Count'].fillna(0) # 沒紀錄就是 0 例

rawdata = pd.merge(rawdata, invest_agg, on=['Week', 'Town'], how='left')
rawdata = pd.merge(rawdata, rt_agg, on=['Week', 'Town'], how='left')
rawdata = pd.merge(rawdata, weather_agg, on=['Week'], how='left') # 氣候只對齊 Week

# 人口密度是對齊「年月」，所以先從 Week 萃取 Year, Month 出來 Join
rawdata['Year'] = rawdata['Week'].dt.year
rawdata['Month'] = rawdata['Week'].dt.month
rawdata = pd.merge(rawdata, pop_agg, on=['Year', 'Month', 'Town'], how='left')

# 排序整理
rawdata = rawdata.sort_values(by=['Town', 'Week']).reset_index(drop=True)

# 調整
rawdata=rawdata[rawdata["Week"]<="2026-01-01"]
rawdata=rawdata[rawdata["Week"]>="2011-01-01"]

# 儲存為 CSV
rawdata.to_csv('Tainan_Dengue_RawData.csv', index=False)
print("合併完成！資料已儲存為 'Tainan_Dengue_RawData.csv'")
print(f"rawdata 資料筆數: {rawdata.shape[0]}, 欄位數: {rawdata.shape[1]}")
rawdata.head()
