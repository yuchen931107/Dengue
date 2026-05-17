import pandas as pd
import requests
import urllib3
import re
import os

def get_population():
    # 1. 關閉 SSL 安全憑證警告
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    url = "https://w3fs.tainan.gov.tw/Download.ashx?u=LzAwMi9GUzAxLzE2L3JlbGZpbGUvOTg0NS84MTU3LzBlNGRlOTNjLWQ3ZGMtNGUzMC05N2I3LTUzYWY2MmE2MGNlZC54bHM%3d&n=MDHoh7rljZfluILlkITljYDph4zmlbjjgIHphLDmlbjjgIHmiLbmlbjjgIHkurrlj6PmlbjjgIHmgKfliKXmr5Tkvovlj4rkurrlj6Plr4bluqbntbHoqIjooaggLnhscw%3d%3d"
    excel_filename = "tainan_population_raw.xls"
    output_csv = "tainan_population_final.csv"
    
    #優化一：檢查本地是否有檔案，避免重覆下載浪費網路時間
    if not os.path.exists(excel_filename):
        print("本地無緩存，正在從網路下載原始 Excel 大檔案...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, verify=False)
            with open(excel_filename, "wb") as f:
                f.write(response.content)
            print("下載成功並已儲存到本地！")
        except Exception as e:
            print(f"下載失敗：{e}")
            exit()
    else:
        print("偵測到本地已存在 Excel 原始檔，自動跳過下載，開啟高速解析模式...")
    
    # 2. 高速精準解析與合併
    try:
        print("\n正在一次性載入所有工作表至記憶體（請稍候幾秒）...")
        
        #優化二核心：sheet_name=None，只開檔一次，全量載入！
        all_sheets_dict = pd.read_excel(excel_filename, sheet_name=None, header=None)
        
        print(f"成功載入！共偵測到 {len(all_sheets_dict)} 個月份的工作表。開始記憶體高速清洗...")
        all_sheets_data = []
        
        # 直接在記憶體字典中進行迴圈，速度極快
        for sheet, df in all_sheets_dict.items():
            
            # 擷取年月
            ym_match = re.match(r'^([\d\.]+)', sheet.strip())
            year_month = ym_match.group(1) if ym_match else sheet
            
            # 清理第一欄文字的空格、換行與特殊空白字元 \xa0
            df[0] = df[0].astype(str).str.replace(r'\s+', '', regex=True).str.replace('\xa0', '', regex=False)
            
            # 根據資料特徵過濾出有效列
            valid_rows = df[df[0].str.endswith('區') | (df[0] == '總計')].copy()
            
            if valid_rows.empty:
                continue
                
            # 提取特定欄位（0: 區域別, 6: 人口數, 10: 人口密度）
            cleaned_df = valid_rows[[0, 6, 10]].copy()
            cleaned_df.columns = ['區域別', '人口數', '人口密度']
            cleaned_df['年月'] = year_month
            
            all_sheets_data.append(cleaned_df)
            
        # 3. 合併與匯出
        if all_sheets_data:
            merged_df = pd.concat(all_sheets_data, ignore_index=True)
            
            # 調整欄位順序
            merged_df = merged_df[['年月', '區域別', '人口數', '人口密度']]
            
            # 強制轉為數值型態
            merged_df['人口數'] = pd.to_numeric(merged_df['人口數'], errors='coerce')
            merged_df['人口密度'] = pd.to_numeric(merged_df['人口密度'], errors='coerce')
            
            # 輸出 CSV
            merged_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"\n【大功告成】秒級高速合併完成！輸出檔案：{output_csv}")
        else:
            print("合併失敗，未找到任何有效的數據列。")
            
    except Exception as e:
        print(f"處理資料時發生錯誤：{e}")
