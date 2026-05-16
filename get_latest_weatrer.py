import requests
import pandas as pd

def get_weather(lat, lon, start_date, end_date):
    print(f"正在下載從 {start_date} 到 {end_date} 的氣象資料...")
    #https://open-meteo.com/
    #從Open-Meteo Api 抓資料
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"#經緯度
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum"
        f"&timezone=Asia%2FTaipei"#時區
    )
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        daily_data = data['daily']
        df_weather = pd.DataFrame({
            'Date': daily_data['time'],
            'Temp_Mean': daily_data['temperature_2m_mean'],
            'Temp_Max': daily_data['temperature_2m_max'],
            'Temp_Min': daily_data['temperature_2m_min'],
            'Rainfall_Sum': daily_data['precipitation_sum']
        })
        
        #確保日期欄位的格式正確
        df_weather['Date'] = pd.to_datetime(df_weather['Date'])
        df_weather.to_csv("Tainan_Weather_Latest.csv", index=False)
        
        print("氣象資料下載成功！")
        print(f"總共取得 {len(df_weather):,} 天的資料")
        print("-" * 40)
        print(df_weather.head())
        
        return df_weather
    
    except Exception as e:
        print(f"Error: {e}")


