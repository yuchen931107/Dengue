import pandas as pd
import urllib.parse
import numpy as np
from scipy.stats import gamma
import epyestim.estimate_r

def count_RT():
    raw_url = "https://github.com/yuchen931107/Dengue/raw/refs/heads/data-storage/Tainan_cases_data.csv"
    safe_url = urllib.parse.quote(raw_url, safe=':/?=')
    
    df = pd.read_csv(safe_url)
    print("資料載入成功，基本資訊如下：")
    df.info()
    
    # 2. 【核心欄位設定】
    # ==========================================
    date_col = '發病日'         
    district_col = '居住鄉鎮' 
    
    # 轉換時間格式並剔除缺漏值
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, district_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # 3. 設定登革熱世代間隔 (Serial Interval) 參數
    # ==========================================
    si_mean, si_std = 15.0, 4.0
    shape = (si_mean / si_std) ** 2
    scale = (si_std ** 2) / si_mean
    si_dist = gamma.pdf(np.arange(1, 31), a=shape, scale=scale)
    si_dist = si_dist / si_dist.sum()
    
    global_start = df[date_col].min()
    global_end = df[date_col].max()
    all_days = pd.date_range(start=global_start, end=global_end, freq='D')
    
    # 4. 定義單一行政區的 Rt 計算函數
    # ==========================================
    def calculate_daily_district_rt(sub_df, si_distribution, total_timeline):
        sub_df = sub_df.sort_values(by=date_col)
        daily_count = sub_df.groupby(date_col).size().reset_index(name='I')
        
        daily_series = daily_count.set_index(date_col).reindex(total_timeline, fill_value=0)['I']
        
        res_raw = epyestim.estimate_r.estimate_r(
            daily_series, 
            si_distribution, 
            0.01, 1, # a_prior, b_prior
            7     # window_size (一週滾動視窗)
        )
        
        res = pd.DataFrame()
        res['Mean(R)'] = res_raw['a_posterior'] * res_raw['b_posterior']
        res['Quantile.0.025(R)'] = epyestim.estimate_r.gamma_quantiles(0.025, res_raw['a_posterior'], res_raw['b_posterior'])
        res['Quantile.0.975(R)'] = epyestim.estimate_r.gamma_quantiles(0.975, res_raw['a_posterior'], res_raw['b_posterior'])
        
        res['Date'] = total_timeline[-len(res):]
        return res
    
    # 5. 核心：用 Groupby 迴圈自動跑遍每一個行政區
    # ==========================================
    all_results = []
    
    for district_name, district_data in df.groupby(district_col):
        if len(district_data) < 10:
            print(f"{district_name} 總病例數過少 ({len(district_data)} 例)，跳過計算。")
            continue
            
        print(f"正在計算 {district_name} 的每日 Rt 趨勢...")
        try:
            district_rt_df = calculate_daily_district_rt(district_data, si_dist, all_days)
            district_rt_df['District'] = district_name
            all_results.append(district_rt_df)
        except Exception as e:
            print(f"{district_name} 計算失敗: {e}")
    
    final_df = pd.concat(all_results, ignore_index=True)
    final_df = final_df[['Date', 'District', 'Mean(R)', 'Quantile.0.025(R)', 'Quantile.0.975(R)']]
    
    # 5.關鍵需求：篩選 2011 到 2025 的資料來畫圖與儲存
    # ==========================================
    final_df_filtered = final_df[
        (final_df['Date'] >= '2011-01-01') & 
        (final_df['Date'] <= '2025-12-29')
    ].copy()
    
    print("\n正在將每日 Rt 轉換為每週平均 Rt...")
    
    all_weekly_results = []
    for district_name, dist_df in final_df_filtered.groupby('District'):
        dist_df_indexed = dist_df.set_index('Date')
        weekly_dist_df = dist_df_indexed[['Mean(R)', 'Quantile.0.025(R)', 'Quantile.0.975(R)']].resample('W-MON').mean()
        weekly_dist_df['District'] = district_name
        weekly_dist_df = weekly_dist_df.reset_index()
        all_weekly_results.append(weekly_dist_df)
    
    calculated_weekly_df = pd.concat(all_weekly_results, ignore_index=True)
    
    # 7. 核心功能：強行補齊台南 37 區
    # ==========================================
    print("\n正在執行特徵工程防禦：比對台南 37 行政區並全面自動補 0...")
    
    tainan_37_districts = [
        '中西區', '東區', '南區', '北區', '安平區', '安南區', 
        '永康區', '歸仁區', '新化區', '左鎮區', '玉井區', '楠西區', 
        '南化區', '仁德區', '關廟區', '龍崎區', '官田區', '麻豆區', 
        '佳里區', '西港區', '七股區', '將軍區', '學甲區', '北門區', 
        '新營區', '後壁區', '白河區', '東山區', '六甲區', '下營區', 
        '柳營區', '鹽水區', '善化區', '大內區', '山上區', '新市區', '安定區'
    ]
    
    unique_weeks = calculated_weekly_df['Date'].unique()
    grid_index = pd.MultiIndex.from_product([unique_weeks, tainan_37_districts], names=['Date', 'District'])
    perfect_grid_df = pd.DataFrame(index=grid_index).reset_index()
    
    final_weekly_merged = pd.merge(perfect_grid_df, calculated_weekly_df, on=['Date', 'District'], how='left')
    
    missing_before = set(tainan_37_districts) - set(calculated_weekly_df['District'].unique())
    print(f"   - 偵測到缺失區域：{list(missing_before)}")
    print("   - 正在將缺失區域與零星空值全面填補為 0.0 ...")
    
    final_weekly_merged['Mean(R)'] = final_weekly_merged['Mean(R)'].fillna(0.0)
    final_weekly_merged['Quantile.0.025(R)'] = final_weekly_merged['Quantile.0.025(R)'].fillna(0.0)
    final_weekly_merged['Quantile.0.975(R)'] = final_weekly_merged['Quantile.0.975(R)'].fillna(0.0)
    final_weekly_merged.rename(columns={'date':'Week'}, inplace=True)
    final_weekly_merged = final_weekly_merged.sort_values(by=['District', 'Date']).reset_index(drop=True)
    final_weekly_merged.to_csv("Tainan_RT.csv", index=False, encoding='utf-8-sig')
    print("處理完成，已輸出 Tainan_RT.csv")
    
    return final_weekly_merged 
