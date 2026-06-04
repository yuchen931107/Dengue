import os
import requests
import time

def fetch_data(target_url):
    """透過 ScraperAPI 抓取資料的工具"""
    api_key = os.getenv("SCRAPERAPI_KEY")
    payload = {
        'api_key': api_key,
        'url': target_url,
        'country_code': 'tw', 
        'premium': 'true' 
    }
    
    max_retries = 3 # 設定最多重試 3 次
    
    for attempt in range(max_retries):
        try:
            print(f"嘗試第 {attempt + 1} 次透過代理連線...")
            response = requests.get('http://api.scraperapi.com', params=payload, timeout=60)
            
            # 如果 ScraperAPI 回傳 500 等錯誤碼，這行會主動拋出例外，進入 except 區塊
            response.raise_for_status() 
            
            response.encoding = 'utf-8-sig'
            return response # 成功的話就直接回傳資料，結束迴圈
            
        except requests.exceptions.HTTPError as e:
            # 捕捉到 500 等 HTTP 錯誤
            print(f"代理伺服器異常 (狀態碼 {response.status_code})")
            if attempt < max_retries - 1:
                print("等待 5 秒後更換節點重試...")
                time.sleep(5)
            else:
                print("已達最大重試次數，請稍後再試。")
                raise e # 重試 3 次都失敗，才把錯誤丟出來
                
        except requests.exceptions.RequestException as e:
            # 捕捉 Timeout 等其他網路錯誤
            print(f"網路連線錯誤: {e}")
            if attempt < max_retries - 1:
                print("等待 5 秒後重試...")
                time.sleep(5)
            else:
                raise e
