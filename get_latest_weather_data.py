import requests
import pandas as pd
import os
import json
from datetime import datetime

def get_weather():
    # ==========================================
    # 1. 基本設定
    # ==========================================
    API_KEY = os.environ.get("CWA_API_KEY")
    GITHUB_USER = "yuchen931107"
    REPO_NAME = "Dengue"
    CSV_FILENAME = "Tainan_History_Weather_2010_2026.csv"
    
    RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/data-storage/{CSV_FILENAME}"
    
    # ==========================================
    # 2. 讀取現有歷史大表
    # ==========================================
    try:
        print("嘗試從雲端讀取歷史大表...")
        df_history = pd.read_csv(RAW_URL)
        df_history['日期'] = pd.to_datetime(df_history['日期'])
        print(f"成功讀取雲端歷史資料，目前共有 {len(df_history)} 筆紀錄。")
    except Exception as e:
        print("雲端尚未有檔案或讀取失敗，嘗試讀取本機檔案...")
        if os.path.exists(CSV_FILENAME):
            df_history = pd.read_csv(CSV_FILENAME)
            df_history['日期'] = pd.to_datetime(df_history['日期'])
        else:
            print("找不到任何歷史基底檔案，將建立全新資料集。")
            df_history = pd.DataFrame()

    #清理歷史資料中的錯誤值並插補
    if not df_history.empty:
        # 定義各欄位的錯誤值門檻
        error_thresholds = {
            '平均相對溼度(%)': -999,
            '日照時數(小時)':  -9,
        }

        for col, threshold in error_thresholds.items():
            if col not in df_history.columns:
                continue

            before = (df_history[col] < threshold).sum()
            if before == 0:
                continue

            # 先將錯誤值替換成 NaN，再依時間序插值
            df_history[col] = df_history[col].where(df_history[col] >= threshold, other=float('nan'))

            # 線性插值（limit_direction 確保首尾也能被 forward/backward fill 補到）
            df_history[col] = df_history[col].interpolate(method='linear', limit_direction='both')

            # 萬一首尾仍有 NaN（整段連續缺測），再用前後填補保底
            df_history[col] = df_history[col].ffill().bfill()

            after = df_history[col].isna().sum()
            print(f"[{col}] 修正 {before} 筆錯誤值，剩餘 NaN：{after} 筆")

    # ==========================================
    # 3. 呼叫「過去歷史觀測 API」
    # ==========================================
    url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/C-B0024-001?Authorization={API_KEY}&StationName=臺南"
    
    print("正在向氣象署請求過去 30 天的完整歷史序列資料...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        
        records = data.get('records', {})
        stations = records.get('station', records.get('Station', records.get('location', [])))
        
        if not stations:
            print("找不到測站欄位，API 實際的根目錄包含：", records.keys())
            return
            
        hourly_data = []
        
        def safe_float(val, is_precip=False):
            try:
                v = float(val)
                if v < 0:
                    return 0.0 if is_precip else float('nan')
                return v
            except (ValueError, TypeError):
                return float('nan')

        for station in stations:
            obs_times = station.get('stationObsTimes', {}).get('stationObsTime', [])
            if not obs_times:
                obs_times = station.get('StationObsTimes', {}).get('StationObsTime', [])
            if not obs_times:
                obs_times = station.get('obsTimes', {}).get('obsTime', [])
                
            for obs in obs_times:
                dt_str = obs.get('dataTime', obs.get('DateTime', obs.get('ObsTime', '')))
                if not dt_str: continue

                elements = obs.get('weatherElements', obs.get('WeatherElements', {}))
                
                if isinstance(elements, dict) and 'weatherElement' in elements:
                    elements = elements['weatherElement']
                elif isinstance(elements, dict) and 'WeatherElement' in elements:
                    elements = elements['WeatherElement']

                temp = precip = rh = sunshine = float('nan')

                if isinstance(elements, dict):
                    temp = safe_float(elements.get('AirTemperature', -99))
                    precip = safe_float(elements.get('Precipitation', -99), is_precip=True)
                    rh = safe_float(elements.get('RelativeHumidity', -99))
                    sunshine = safe_float(elements.get('SunshineDuration', -99))
                
                elif isinstance(elements, list):
                    for el in elements:
                        name = el.get('elementName', el.get('ElementName', ''))
                        val = el.get('elementValue', el.get('ElementValue', '-99'))
                        
                        if name == 'AirTemperature': temp = safe_float(val)
                        elif name == 'Precipitation': precip = safe_float(val, is_precip=True)
                        elif name == 'RelativeHumidity': rh = safe_float(val)
                        elif name == 'SunshineDuration': sunshine = safe_float(val)

                hourly_data.append({
                    "時間": pd.to_datetime(dt_str),
                    "氣溫": temp,
                    "降水量": precip,
                    "相對溼度": rh,
                    "日照時數": sunshine,
                    "有降水旗標": 1 if precip > 0 else 0
                })
                
        if not hourly_data:
            print("無法從 JSON 中萃取出所需欄位。以下為氣象署 API 回傳的真實第一筆結構：")
            print(json.dumps(stations[0] if stations else data, indent=2, ensure_ascii=False)[:1000])
            return
            
        df_api_hourly = pd.DataFrame(hourly_data)
        df_api_hourly['時間'] = df_api_hourly['時間'].dt.tz_localize(None)
        
        # ==========================================
        # 4. 聚合成日資料
        # ==========================================
        print("資料解析完畢，正在結算完美的每日最高/低溫與累積量...")
        df_api_hourly['日期'] = df_api_hourly['時間'].dt.normalize()
        
        df_api_daily = df_api_hourly.groupby('日期').agg(
            平均氣溫=('氣溫', 'mean'),
            最高氣溫=('氣溫', 'max'),
            最低氣溫=('氣溫', 'min'),
            日累積降水量=('降水量', 'sum'), 
            平均相對溼度=('相對溼度', 'mean'),
            日照時數=('日照時數', 'sum'),
            降水時數=('有降水旗標', 'sum')
        ).reset_index()
        
        df_api_daily['溫差(℃)'] = df_api_daily['最高氣溫'] - df_api_daily['最低氣溫']
        
        df_api_daily = df_api_daily.rename(columns={
            '平均氣溫': '平均氣溫(℃)',
            '日累積降水量': '日累積降水量(mm)',
            '平均相對溼度': '平均相對溼度(%)',
            '日照時數': '日照時數(小時)',
            '降水時數': '降水時數(小時)'
        })
        
        for col in ['平均氣溫(℃)', '溫差(℃)', '平均相對溼度(%)', '日照時數(小時)']:
            df_api_daily[col] = df_api_daily[col].round(1)
            
        target_cols = ['日期', '平均氣溫(℃)', '溫差(℃)', '平均相對溼度(%)', '日照時數(小時)', '降水時數(小時)', '日累積降水量(mm)']
        df_api_daily = df_api_daily[target_cols]
        
        # ==========================================
        # 5. 新舊資料無縫接軌與覆蓋去重
        # ==========================================
        if not df_history.empty:
            df_final = pd.concat([df_history, df_api_daily], ignore_index=True)
        else:
            df_final = df_api_daily
            
        df_final = df_final.drop_duplicates(subset=['日期'], keep='last')
        df_final = df_final.sort_values(by='日期').reset_index(drop=True)
        
        # ==========================================
        # 6. 輸出更新
        # ==========================================
        df_final.to_csv(CSV_FILENAME, index=False, encoding="utf-8-sig")
        print(f"日資料更新成功！最新資料已推進至：{df_final['日期'].max().strftime('%Y-%m-%d')}")
        
        return df_final
        
    else:
        print(f"API 連線失敗，錯誤碼: {response.status_code}")
