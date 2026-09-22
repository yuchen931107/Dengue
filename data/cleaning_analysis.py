import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Tainan_Dengue_RawData.csv")

df.info()
print(df['RT_level'].value_counts(dropna=False))
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft JhengHei', '微軟正黑體'] 
plt.rcParams['axes.unicode_minus'] = False 
# ---------------------------------------------------------
# 1. 樣本數 N 與 變數數量
# ---------------------------------------------------------
n_samples, n_variables = df.shape
print("\n【基本資訊】")
print(f"樣本數 N: {n_samples}")
print(f"變數數量: {n_variables}")
print("\n欄位型態概覽:")
print(df.dtypes)
print("-" * 40)

# ---------------------------------------------------------
# 2. 缺失值檢查 (清洗前)
# ---------------------------------------------------------
print("【缺失值檢查 (清洗前)】")
missing_ratios = (df.isnull().sum() / len(df)) * 100
missing_ratios = missing_ratios[missing_ratios > 0].sort_values(ascending=False)

if not missing_ratios.empty:
    print("各欄位缺失值比例 (%):")
    print(missing_ratios.round(2))
else:
    print("資料無任何缺失值！")
print("-" * 40)

# ---------------------------------------------------------
# 3. 資料清洗 (Data Cleaning)
# ---------------------------------------------------------
print("【開始進行資料清洗...】")
# (1) 確診數：沒紀錄就是 0 確診
df['Case_Count'] = df['Case_Count'].fillna(0)

# (2) 病媒蚊指數：依需求直接補 0
invest_cols = ['BI', 'CI', 'HI', 'LI', 'AI', 'PI', 'Con100HH']
df[invest_cols] = df[invest_cols].fillna(0)

# (3) 噴藥紀錄 (若有遺失值，保險起見補 0)
if 'Spray_Count' in df.columns:
    df['Spray_Count'] = df['Spray_Count'].fillna(0)
if 'Medicine' in df.columns:
    df['Medicine'] = df['Medicine'].fillna(0)

# ---------------------------------------------------------
# 4. 異常值檢查與清洗 (氣候資料)
# ---------------------------------------------------------
main_features = ['AvgTemp','TempRange','AvgHumidity','SunshineHours','RainfallHours','Rainfall']

# 畫清洗前的箱型圖
for feature in main_features:
    plt.figure(figsize=(8, 3)) 
    sns.boxplot(x=df[feature], color='skyblue')
    plt.title(f'異常值檢查 (清洗前)：{feature}', fontsize=14)
    plt.xlabel(f'{feature} 數值')
    plt.tight_layout()
    plt.show()

print("處理氣候異常值...")
# 發現 SunshineHours 及 AvgHumidity 有負值不合理，轉換成 nan 後用前向填充 (ffill)
df.loc[df['AvgHumidity'] < 0, 'AvgHumidity'] = np.nan
df.loc[df['SunshineHours'] < 0, 'SunshineHours'] = np.nan

df['AvgHumidity'] = df['AvgHumidity'].ffill()
df['SunshineHours'] = df['SunshineHours'].ffill()

main_features_proc = ['AvgHumidity','SunshineHours']
for feature in main_features_proc:
    plt.figure(figsize=(8, 3)) 
    sns.boxplot(x=df[feature], color='skyblue')
    plt.title(f'處理後檢查：{feature}', fontsize=14)
    plt.xlabel(f'{feature} 數值')
    plt.tight_layout()
    plt.show()
print("-" * 40)

# ---------------------------------------------------------
# 5. 目標變數分布 
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df.dropna(subset=['RT_level']), x='RT_level', palette='Set2')
plt.title('目標變數 (RT_level) 類別分布圖', fontsize=14)
plt.xlabel('RT_level (等級)')
plt.ylabel('樣本數 (Count)')
plt.show()

# ---------------------------------------------------------
# 6. 主要特徵變數分布
# ---------------------------------------------------------
df[main_features].hist(bins=30, figsize=(10, 8), color='teal', edgecolor='black')
plt.suptitle('主要特徵變數分布', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. 相關性分析 
# ---------------------------------------------------------
plt.figure(figsize=(14, 10))
# 只選取數值型別的欄位計算相關係數 (忽略字串如 Town, Week 等)
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr(method='pearson') 

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
            vmax=1, vmin=-1, center=0, square=True, linewidths=.5)
plt.title('變數相關性熱力圖 (Correlation Matrix)', fontsize=16)
plt.show()

# =========================================================
# 8. 儲存清洗後的資料供機器學習模型 (LSTM) 使用
# =========================================================
df.to_csv("Tainan_Dengue_ML.csv", index=False, encoding='utf-8-sig')
print("資料清洗與分析完成！已輸出機器學習專用檔 'Tainan_Dengue_ML.csv'")