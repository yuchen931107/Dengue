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
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft JhengHei', '微軟正黑體'] 
plt.rcParams['axes.unicode_minus'] = False 
# ---------------------------------------------------------
# 1.樣本數 N 與 變數數量
# ---------------------------------------------------------
n_samples, n_variables = df.shape
print("【基本資訊】")
print(f"樣本數 N: {n_samples}")
print(f"變數數量: {n_variables}")
print("\n欄位型態概覽:")
print(df.dtypes)
print("-" * 40)

# ---------------------------------------------------------
# 2.缺失值比例
# ---------------------------------------------------------
print("【缺失值檢查】")
missing_ratios = (df.isnull().sum() / len(df)) * 100
missing_ratios = missing_ratios[missing_ratios > 0].sort_values(ascending=False)

if not missing_ratios.empty:
    print("各欄位缺失值比例 (%):")
    print(missing_ratios.round(2))
else:
    print("資料無任何缺失值！")
print("-" * 40)

# ---------------------------------------------------------
# 3.異常值檢查 (利用箱型圖 Boxplot)
# ---------------------------------------------------------
# 這裡挑選幾個關鍵的數值特徵作為代表 (氣候)
main_features = ['AvgTemp','TempRange','AvgHumidity','SunshineHours',
                 'RainfallHours','Rainfall']

# 使用迴圈，為每一個特徵獨立畫一張箱型圖
for feature in main_features:
    plt.figure(figsize=(8, 3)) 
    sns.boxplot(x=df[feature], color='skyblue')
    plt.title(f'異常值檢查：{feature}', fontsize=14)
    plt.xlabel(f'{feature} 數值')
    plt.tight_layout()
    plt.show()

'''
發現SunshineHours及AvgHumidity有負值不合理 應為遺失代碼故將其轉換成nan
'''
df.loc[df['AvgHumidity'] < 0, 'AvgHumidity'] = np.nan
df.loc[df['SunshineHours'] < 0, 'SunshineHours'] = np.nan
main_features_proc = ['AvgHumidity','SunshineHours']

for feature in main_features_proc:
    plt.figure(figsize=(8, 3)) 
    sns.boxplot(x=df[feature], color='skyblue')
    plt.title(f'異常值檢查：{feature}', fontsize=14)
    plt.xlabel(f'{feature} 數值')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 4.目標變數分布 
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
# 加上 dropna() 確保如果目標變數有缺失值不會報錯
sns.countplot(data=df.dropna(subset=['RT_level']), x='RT_level', palette='Set2')
plt.title('目標變數 (RT_level) 類別分布圖', fontsize=14)
plt.xlabel('RT_level (等級)')
plt.ylabel('樣本數 (Count)')
plt.show()

# ---------------------------------------------------------
# 5.主要特徵變數分布
# ---------------------------------------------------------
# 繪製多個直方圖
df[main_features].hist(bins=30, figsize=(10, 8), color='teal', edgecolor='black')
plt.suptitle('主要特徵變數分布', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6.相關性分析 
# ---------------------------------------------------------
plt.figure(figsize=(14, 10))
# 只選取數值型別的欄位計算相關係數 (自動忽略字串如 Town, Week)
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr(method='pearson') 

# 繪製熱力圖，設定 mask 把右上角遮住會更簡潔 (非必須，但視覺較佳)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
            vmax=1, vmin=-1, center=0, square=True, linewidths=.5)
plt.title('變數相關性熱力圖 (Correlation Matrix)', fontsize=16)
plt.show()