import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from RT_split import RT_split
df = pd.read_csv("Tainan_Dengue_RawData.csv")
df = RT_split(df,rt_column='RT', new_column='RT_level')
df=df.drop(columns=["Population density"])
df.info()
print(df['RT_level'].value_counts(dropna=False))

# ==========================================
# 0. 環境設定與讀取資料
# ==========================================
# 設定畫圖風格與支援繁體中文的字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # Windows 適用 (Mac 請改 'PingFang HK')
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="Microsoft JhengHei") 

print("正在讀取資料...")
df['Week'] = pd.to_datetime(df['Week'])

# 新增一個二元標籤「當週是否有疫情」，方便後續比較
df['Has_Case'] = df['Case_Count'].apply(lambda x: '有病例' if x > 0 else '無病例')

# ==========================================
# 1. 缺失值全貌可視化 (Missing Value Analysis)
# ==========================================
print("\n[1/5] 繪製缺失值分佈圖...")
plt.figure(figsize=(10, 6))
# 計算各欄位的缺失值比例
missing_ratio = (df.isnull().sum() / len(df)) * 100
missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
sns.barplot(x=missing_ratio.values, y=missing_ratio.index, palette="Reds_r")
plt.title('各變數缺失值比例 (%) - 保留真實的未調查狀態')
plt.xlabel('缺失比例 (%)')
plt.show()

# ==========================================
# 2. 疫情時空分佈 (Spatio-Temporal Analysis)
# ==========================================
print("\n[2/5] 繪製疫情時空分佈圖...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 2a. 季節性分析 (月份)
monthly_cases = df.groupby('Month')['Case_Count'].sum().reset_index()
sns.barplot(data=monthly_cases, x='Month', y='Case_Count', palette="viridis", ax=axes[0])
axes[0].set_title('各月份歷史累計病例數 (觀察季節性)')

# 2b. 空間熱區分析 (行政區前15名)
town_cases = df.groupby('Town')['Case_Count'].sum().sort_values(ascending=False).head(15).reset_index()
sns.barplot(data=town_cases, y='Town', x='Case_Count', palette="magma", ax=axes[1])
axes[1].set_title('歷史累計病例數 TOP 15 行政區')
plt.tight_layout()
plt.show()

# ==========================================
# 3. 氣候變數與疫情的關係 (Meteorological Analysis)
# ==========================================
print("\n[3/5] 繪製氣候變數分析圖...")
weather_cols = ['AvgTemp', 'AvgHumidity', 'Rainfall']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(weather_cols):
    # 濾除氣候空值，觀察有病例 vs 無病例的氣候差異
    plot_data = df.dropna(subset=[col, 'Has_Case'])
    sns.boxplot(data=plot_data, x='Has_Case', y=col, ax=axes[i], palette="Set2")
    axes[i].set_title(f'{col} 於「有無疫情」時的分佈')

plt.tight_layout()
plt.show()

# ==========================================
# 4. 蟲媒指標與疫情的關係 (Entomological Analysis)
# ==========================================
print("\n[4/5] 繪製蟲媒指標分析圖 (已排除未調查之空值)...")
invest_cols = ['BI', 'CI', 'HI']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(invest_cols):
    # 【關鍵防錯】：只取有去調查 (BI非空值) 的資料來畫圖
    plot_data = df.dropna(subset=[col])
    # 為了避免極端值把圖壓扁，使用對數座標 (log scale)
    sns.boxplot(data=plot_data, x='Has_Case', y=col, ax=axes[i], palette="YlOrRd")
    axes[i].set_yscale('symlog') # 使用 symlog 處理包含 0 的對數縮放
    axes[i].set_ylabel(f'{col} (Log Scale)')
    axes[i].set_title(f'實際有調查的 {col} 指數 vs 疫情')

plt.tight_layout()
plt.show()

# ==========================================
# 5. 變數相關性熱力圖 (Correlation Heatmap)
# ==========================================
print("\n[5/5] 繪製特徵相關性熱力圖...")
plt.figure(figsize=(12, 10))
# 選取數值型欄位
numeric_cols = ['Case_Count', 'RT', 'BI', 'CI', 'HI', 'AvgTemp', 'TempRange', 'AvgHumidity', 'Rainfall', 'Population density']
corr_data = df[numeric_cols].corr(method='spearman') # 使用 spearman 抵抗極端值

# 畫熱力圖
mask = np.triu(np.ones_like(corr_data, dtype=bool)) # 只顯示下半三角
sns.heatmap(corr_data, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
            vmin=-1, vmax=1, square=True, linewidths=.5)
plt.title('各特徵間的 Spearman 相關係數熱力圖 (觀察共線性與預測力)')
plt.show()
print("\n✅ 初步分析全部完成！")
# ==========================================
# ==========================================
# 2. 針對 'Town' (區域) 進行群組化，並計算 'BI' 欄位中非空值的數量
# (因為 BI, CI 等指數通常是同時調查，計算其中一個非空值數量即可代表調查次數)
survey_counts = df.groupby('Town')['BI'].count().reset_index(name='Survey_Times')

# 3. 依照調查次數由高到低進行排序
survey_counts = survey_counts.sort_values(by='Survey_Times', ascending=False)

# 4. 重置索引 (讓表格看起來更整齊)
survey_counts = survey_counts.reset_index(drop=True)

# 顯示前 10 名調查次數最多的區域
print(survey_counts.head(37))
# ==========================================
# 2. 依照「年份 (Year)」進行分組，並加總「病例數 (Case_Count)」
yearly_cases = df.groupby('Year')['Case_Count'].sum().reset_index()

# 確保年份顯示為整數
yearly_cases['Year'] = yearly_cases['Year'].astype(int)

# 3. 繪製長條圖
plt.figure(figsize=(10, 6))
plt.bar(yearly_cases['Year'], yearly_cases['Case_Count'], color='#4C72B0')

# 設定圖表標題與軸標籤
plt.title('Total Dengue Cases in Tainan by Year', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Case Count', fontsize=12)

# 設定 X 軸刻度為各個年份，避免出現小數點
plt.xticks(yearly_cases['Year'])

# 加入水平格線方便對齊閱讀
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 4. 調整排版並顯示/儲存圖表
plt.tight_layout()
plt.savefig("yearly_dengue_cases.png") # 儲存圖表
plt.show() # 顯示圖表

# 5. 印出統計結果的表格
print(yearly_cases.to_markdown(index=False))




