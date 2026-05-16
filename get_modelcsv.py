import pandas as pd
import requests
import io

def get_Mos():
    #https://data.gov.tw/dataset/24159
    csv_url = "https://od.cdc.gov.tw/eic/MosIndex_Tainan.csv" 
    
    #設定Header模擬正常瀏覽器行為，避免被政府網站的防火牆(WAF)擋下
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        print("正在從政府開放資料平台下載最新資料...")
        response = requests.get(csv_url, headers=headers)
        response.raise_for_status()#針測錯誤
        response.encoding = 'utf-8-sig' 
        df = pd.read_csv(io.StringIO(response.text))

        df.to_csv("MosIndex_Tainan_Latest.csv", index=False, encoding='utf-8-sig')
        
        print("資料下載並更新成功！")
        print(f"目前資料總筆數：{len(df):,} 筆")
        print("-" * 40)
        print("預覽前三筆資料：")
        print(df.head(3))
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"網路連線錯誤: {e}")
    except Exception as e:
        print(f"資料處理發生錯誤: {e}")

    
