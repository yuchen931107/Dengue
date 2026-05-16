import pandas as pd
import requests
import io

def get_forecast_data():
    #疾病管制署資料開放平台
    #https://data.cdc.gov.tw/dataset/dengue-daily-determined-cases-1998/resource/e868ae05-2381-44f2-9656-42292ef7e0c6
    url = "https://od.cdc.gov.tw/eic/Dengue_Daily.csv"
    print("正在下載每日確診病例資料...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        response.encoding = 'utf-8-sig'
        
        df = pd.read_csv(io.StringIO(response.text), low_memory=False)
        df.columns = df.columns.str.strip()
        #選取縣市
        df_tainan = df[df['居住縣市'] == '台南市']

        #將發病日轉換為時間格式
        df_tainan['發病日'] = pd.to_datetime(df_tainan['發病日'])
        df_tainan = df_tainan.sort_values('發病日')
        
        df_tainan.to_csv("Tainan_data.csv", index=False)
        
        print(f"成功取得台南市病例資料！共 {len(df_tainan):,} 筆")
        print("最新五筆發病紀錄：")
        print(df_tainan[['發病日', '居住鄉鎮', '居住村里']].tail())
        
        return df_tainan
        
    except Exception as e:
        print(f"發生錯誤: {e}")

    
