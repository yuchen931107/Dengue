import requests
import pandas as pd
import io
import os
from datetime import datetime

# ==========================================
# 1. 基本設定
# ==========================================
API_KEY = "CWA-0908E3D1-9A58-43A9-BBCD-699352344CF7"
GITHUB_USER = "yuchen931107"
REPO_NAME = "dengue-predict"
CSV_FILENAME = "Tainan_History_Weather_2010_2026.csv"

# 雲端 data-storage 分支的原始資料網址
RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/data-storage/{CSV_FILENAME}"

# ==========================================
# 2. 讀取現有的歷史大表 (基底)
# ==========================================
try:
    print("⏳ 嘗試從雲端 data-storage 分支讀取現有歷史大表...")
    df_history = pd.read_csv(RAW_URL)
    df_history['日期'] = pd.to_datetime(df_history['日期'])
    print(f"✅ 成功讀取雲端歷史資料，目前共有 {len(df_history)} 筆紀錄。")
except Exception as e:
    print("⚠️ 雲端尚未有檔案或讀取失敗，嘗試讀取本機同名檔案...")
    if os.path.exists(CSV_FILENAME):
        df_history = pd.read_csv(CSV_FILENAME)
        df_history['日期'] = pd.to_datetime(df_history['日期'])
        print(f"✅ 成功讀取本機檔案，目前共有 {len(df_history)} 筆紀錄。")
    else:
        print("❌ 錯誤：找不到任何歷史基底檔案，請確保檔案存在！")
        raise e

# ==========================================
# 3. 透過 API 撈取最近幾天的最新氣象 (觸角)
# ==========================================
# 設定初始日期為歷史大表的最後一天的前兩天 (防禦性補漏洞，避免Actions某天漏跑)
start_date = df_history['日期'].max().strftime('%Y-%m-%d')
time_from_param = f"{start_date}T00:00:00"

url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization={API_KEY}&StationName=臺南&timeFrom={time_from_param}"

print(f"⏳ 正在向氣象署 API 請求自 {start_date} 以來的最新每小時觀測資料...")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    station_data = data['records']['Station'][0]
    obs_records = station_data['StationObsTimes']['StationObsTime']
    
    hourly_data = []
    for record in obs_records:
        obs_time = record['DateTime']
        elements = record['WeatherElement']
        
        row = {
            "時間": pd.to_datetime(obs_time),
            "平均氣溫(℃)": float(elements.get('AirTemperature', 0.0)),
            "平均相對溼度(%)": float(elements.get('RelativeHumidity', 0.0)),
            "日累積降水量(mm)": float(elements.get('Now', {}).get('Precipitation', 0.0))
        }
        hourly_data = []
        hourly_data.append(row)
        
    df_api_hourly = pd.DataFrame(hourly_data)
    
    # ==========================================
    # 4. 關鍵：將每小時資料聚合(Aggregate)成日資料，對齊歷史大表
    # ==========================================
    print("🧹 正在將 API 每小時資料轉換為日資料格式...")
    df_api_hourly['日期'] = df_api_hourly['時間'].dt.normalize() # 將時間去掉只留日期
    
    # 算每日摘要：平均溫、最高溫、最低溫、平均濕度、當天最大累積雨量
    df_api_daily = df_api_hourly.groupby('日期').agg({
        '平均氣溫(℃)': 'mean',
        '平均相對溼度(%)': 'mean',
        '日累積降水量(mm)': 'max' # 氣象署該標籤本身就是當天累積，取 max 即為全天總雨量
    }).reset_index()
    
    # 補上歷史大表需要的其他欄位（API不支援或不重要的欄位補 NaN，確保維度一致）
    df_api_daily['最高氣溫(℃)'] = df_api_hourly.groupby('日期')['平均氣溫(℃)'].transform('max')
    df_api_daily['最低氣溫(℃)'] = df_api_hourly.groupby('日期')['平均氣溫(℃)'].transform('min')
    df_api_daily['降水時數(小時)'] = pd.NA
    df_api_daily['日照時數(小時)'] = pd.NA
    df_api_daily['平均風速(m/s)'] = pd.NA
    
    # 調整欄位順序，使其跟歷史大表完全一模一樣
    df_api_daily = df_api_daily[df_history.columns]
    
    # ==========================================
    # 5. 新舊資料橫向黏合與去重
    # ==========================================
    print("🔗 正在合併新舊資料序列並進行去重...")
    df_final = pd.concat([df_history, df_api_daily], ignore_index=True)
    
    # 根據「日期」欄位去重，若有重疊，保留最新寫入的 API 資料（keep='last'）
    df_final = df_final.drop_duplicates(subset=['日期'], keep='last')
    
    # 排序
    df_final = df_final.sort_values(by='日期').reset_index(drop=True)
    
    # ==========================================
    # 6. 儲存結果
    # ==========================================
    df_final.to_csv(CSV_FILENAME, index=False, encoding="utf-8-sig")
    print(f"🎉 整合成功！資料表已更新。最新日期為：{df_final['日期'].max().strftime('%Y-%m-%d')}，總筆數：{len(df_final)}")

else:
    print(f"❌ API 連線失敗，狀態碼: {response.status_code}，本日未更新。")
