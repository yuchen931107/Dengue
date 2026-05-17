import pandas as pd
import io
import os

#1.設定你要合併的年份區間
start_year = 2010
end_year = 2026

#2.定義對齊氣象署格式的「9格一組」寬度陣列
widths = [6, 1, 8] + [9] * 40

#3.完整代號對齊對應表
columns_mapping = [
    "站碼", "space", "日期", 
    "PS01", "PS02", "PS03", "PS04", "PS05", "PS06", "TX01", 
    "TX02", "TX03", "TX04", "TX05", "TD01", "RH01", "RH04", "RH05", 
    "WD01", "WD02", "WD07", "WD08", "WD09", "PP01", "PP02", "PP03", 
    "PP04", "PP05", "PP06", "SS01", "SS02", "GR01", "VS01", "EP05", 
    "UV01", "UV03", "CD11", "TS01", "TS02", "TS03", "TS04", "TS05", "TS06"
]

#中文化對照表
chinese_columns = {
    "測站代碼": "測站代碼",
    "日期": "日期",
    "平均氣溫(℃)": "平均氣溫(℃)",
    "最高氣溫(℃)": "最高氣溫(℃)",
    "最低氣溫(℃)": "最低氣溫(℃)",
    "平均相對溼度(%)": "平均相對溼度(%)",
    "日累積降水量(mm)": "日累積降水量(mm)",
    "降水時數(小時)": "降水時數(小時)",
    "日照時數(小時)": "日照時數(小時)",
    "平均風速(m/s)": "平均風速(m/s)"
}
raw_keys = ["站碼", "日期", "TX01", "TX02", "TX04", "RH01", "PP01", "PP02", "SS01", "WD01"]

# 4. 準備一個空清單，用來存放每年清洗好的 DataFrame
all_years_data = []

print("開始執行全年份氣象資料批次解碼與合併 Pipeline...")

#5.用迴圈走訪 2010 到 2026 年
for year in range(start_year, end_year + 1):
    # 根據你的命名規則動態組合出檔名
    file_name = f"LotsDataReports_{year}.txt"
    
    #防呆機制：檢查檔案到底在不在，免得其中一年漏下載導致程式崩潰
    if not os.path.exists(file_name):
        print(f"找不到檔案 {file_name}，自動跳過該年份。")
        continue
        
    print(f"正在處理：{file_name} ...")
    
    cleaned_lines = []
    temp_line = ""
    
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("*") or line.startswith("#") or not line.strip():
                continue
            if line.strip().startswith("467410"):
                if temp_line:
                    cleaned_lines.append(temp_line)
                temp_line = line.replace("\n", "")
            else:
                temp_line += line.replace("\n", "")
        if temp_line:
            cleaned_lines.append(temp_line)
            
    #解析格子
    joined_text = "\n".join(cleaned_lines)
    df_year = pd.read_fwf(io.StringIO(joined_text), widths=widths, header=None)
    df_year.columns = columns_mapping[:df_year.shape[1]]
    
    #篩選出核心欄位並重新命名
    df_year_filtered = df_year[raw_keys].copy()
    df_year_filtered.columns = list(chinese_columns.values())
    
    #將這一年的資料塞進大清單
    all_years_data.append(df_year_filtered)

#6.如果清單不為空，將所有年份的 DataFrame 直向黏起來！
if all_years_data:
    df_all_combined = pd.concat(all_years_data, ignore_index=True)
    
    #7.進行整張大表的資料清洗
    print("正在進行全表最終資料清洗...")
    df_all_combined["日期"] = pd.to_datetime(df_all_combined["日期"].astype(str), format="%Y%m%d")
    
    #處理特殊值 (雨量 None / -9.8 轉 0.0)
    df_all_combined["日累積降水量(mm)"] = (
        df_all_combined["日累積降水量(mm)"]
        .astype(str)
        .str.strip()
        .replace({"None": "0.0", "-9.8": "0.0"})
        .astype(float)
    )
    
    #確保數值欄位全部都是數字格式
    num_cols = ["平均氣溫(℃)", "最高氣溫(℃)", "最低氣溫(℃)", "平均相對溼度(%)", "降水時數(小時)", "日照時數(小時)", "平均風速(m/s)"]
    for col in num_cols:
        df_all_combined[col] = pd.to_numeric(df_all_combined[col], errors="coerce")
        
    #依照日期排序，確保時間序列正確
    df_all_combined = df_all_combined.sort_values(by="日期").reset_index(drop=True)
    df_all_combined = df_all_combined[['日期','平均氣溫(℃)','日累積降水量(mm)']]
    
    print("\n全年份歷史資料合併成功！")
    print(f"總資料筆數（天數）：{len(df_all_combined)} 筆")
    print(f"時間區間：{df_all_combined['日期'].min().strftime('%Y-%m-%d')} ~ {df_all_combined['日期'].max().strftime('%Y-%m-%d')}")
    
    # 8. 匯出成最終版歷史氣象大表
    output_filename = "Tainan_History_Weather_2010_2026.csv"
    df_all_combined.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"歷史大基底已成功存為：{output_filename}")
    
else:
    print("合併失敗，請檢查檔案名稱或路徑是否正確。")
df_all_combined.info()

