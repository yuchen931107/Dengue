import os
os.chdir(r'C:\Users\User\Desktop\Dengue\Dengue-main')
import pandas as pd
import urllib.parse
from Dengue_level import level

raw_url = "https://raw.githubusercontent.com/yuchen931107/Dengue/refs/heads/data-storage/Tainan_Dengue_ML.csv"
safe_url = urllib.parse.quote(raw_url, safe=':/?=')
df = pd.read_csv(safe_url)
df = level(df,case_column='Case_Count', new_column='Dengue_level')
df.info()

#==========================================================
#基本統計量
print("\n=== 基本統計量 ===")
print(df.describe())
stat=df.describe()

#檢查是否有缺失值
print("\n=== 缺失值統計 ===")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0]) #只顯示有缺失值的欄位
#==========================================================
import matplotlib.pyplot as plt
import seaborn as sns

#解決matplotlib 中文顯示問題 (如果你在 Colab 執行，可能需要額外安裝字型)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # Windows 使用微軟正黑體
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac 使用者可改用此行
plt.rcParams['axes.unicode_minus'] = False # 確保負號顯示正常

# 繪製所有數值特徵的直方圖
df.hist(bins=20, figsize=(15, 10), color='steelblue', edgecolor='black')
plt.suptitle("各數值變數分布狀態", fontsize=16)
plt.tight_layout()
plt.show()
#==========================================================
plt.figure(figsize=(12, 10))

# 只篩選數值型欄位計算相關係數
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()

# 繪製熱力圖
sns.heatmap(corr_matrix, 
            annot=True,       # 顯示數值
            cmap='coolwarm',  # 顏色越紅代表正相關越強，越藍代表負相關越強
            fmt=".2f",        # 小數點後兩位
            linewidths=0.5)

plt.title("變數相關性熱力圖", fontsize=16)
plt.show()
#==========================================================
target_col = 'Case_Count' 

if target_col in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target_col, palette='Set2')
    plt.title(f"目標變數 ({target_col}) 分布狀況")
    plt.show()
    
    # 顯示具體比例
    print(df[target_col].value_counts(normalize=True))
else:
    print("請替換 target_col 為實際的目標變數名稱。")
