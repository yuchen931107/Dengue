import requests
import pandas as pd
import io
import os
from datetime import datetime

def get_weather():
    # ==========================================
    # 1. 基本設定
    # ==========================================
    API_KEY = os.environ.get("CWA_API_KEY")
    GITHUB_USER = "yuchen931107"
    REPO_NAME = "Dengue"
    CSV_FILENAME = "Tainan_History_Weather_2010_2026.csv"
    
    # 雲端 data-storage 分支的原始資料網址
    RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/data-storage/{CSV_FILENAME}"
    
    # ==========================================
    # 2. 讀取現有的極簡歷史大表 (基底)
    # ==========================================
    try:
        print("嘗試從雲端 data-storage 分支讀取現有歷史大表...")
        df_history = pd.read_csv(RAW_URL)
        df_history['日期'] = pd.to_datetime(df_history['日期'])
        print(f"成功讀取雲端歷史資料，目前共有 {len(df_history)} 筆紀錄。")
    except Exception as e:
        print("雲端尚未有檔案或讀取失敗，嘗試讀取本機同名檔案...")
        if os.path.exists(CSV_FILENAME):
            df_history = pd.read_csv(CSV_FILENAME)
            df_history['日期'] = pd.to_datetime(df_history['日期'])
        else:
            print("錯誤：找不到任何歷史基底檔案！")
            raise e
    
    # ==========================================
    # 3. 透過 API 撈取【過去 24 小時】最新氣象
    # ==========================================
    url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0003-001?Authorization={API_KEY}&StationName=臺南"
    
    print("正在向氣象署 API 請求過去 24 小時的完整每小時序列資料...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        stations = data['records']['Station']
        hourly_data = []
        
        # 用扁平化迴圈直接走訪每一個小時的測站物件
        for station_data in stations:
            obs_time = station_data['ObsTime']['DateTime']
            elements = station_data['WeatherElement']
            
            row = {
                "時間": pd.to_datetime(obs_time),
                "平均氣溫(℃)": float(elements.get('AirTemperature', 0.0)),
                "日累積降水量(mm)": float(elements.get('Now', {}).get('Precipitation', 0.0))
            }
            hourly_data.append(row)
            
        df_api_hourly = pd.DataFrame(hourly_data)
        df_api_hourly['時間'] = df_api_hourly['時間'].dt.tz_localize(None) # 消除時區衝突
        
        # ==========================================
        # 4. 將每小時資料聚合成日資料
        # ==========================================
        print(" 正在將 API 每小時資料轉換為日資料格式...")
        df_api_hourly['日期'] = df_api_hourly['時間'].dt.normalize()
        
        df_api_daily = df_api_hourly.groupby('日期').agg({
            '平均氣溫(℃)': 'mean',
            '日累積降水量(mm)': 'max' 
        }).reset_index()
        
        # 自動適應極簡三欄位 [日期, 平均氣溫(℃), 日累積降水量(mm)]
        df_api_daily = df_api_daily[df_history.columns]
        
        # ==========================================
        # 5. 新舊資料橫向黏合與去重
        # ==========================================
        print("正在合併新舊資料序列並進行去重...")
        df_final = pd.concat([df_history, df_api_daily], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=['日期'], keep='last')
        df_final = df_final.sort_values(by='日期').reset_index(drop=True)
        
        # ==========================================
        # 6. 儲存結果
        # ==========================================
        df_final.to_csv(CSV_FILENAME, index=False, encoding="utf-8-sig")
        print(f"整合成功！最新日期已自動無縫補齊至：{df_final['日期'].max().strftime('%Y-%m-%d')}，總筆數：{len(df_final)}")
    
    else:
        print(f"API 連線失敗，狀態碼: {response.status_code}")
