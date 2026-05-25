import pandas as pd
import numpy as np
def level(df, case_column='Case_Count', new_column='Dengue_level'):
    """
    將連續病例數切分為 4 個預警等級：
    Level 0: 0 例
    Level 1: 1-4 例 (散發)
    Level 2: 5-49 例 (局部群聚)
    Level 3: >= 50 例 (大規模)
    """
    bins = [-np.inf, 0, 4, 49, np.inf]
    labels = [0, 1, 2, 3] 
    df[new_column] = pd.cut(df[case_column], bins=bins, labels=labels)
    return df
'''
# ---------------------------------------------------------
# 驗證步驟：檢查各等級的資料佔比是否符合預期
# ---------------------------------------------------------
print("=== 4 級分類轉換結果驗證 ===")
counts = df['Dengue_level'].value_counts().sort_index()
percentages = df['Dengue_level'].value_counts(normalize=True).sort_index() * 100
validation_df = pd.DataFrame({
    '筆數': counts,
    '佔比 (%)': percentages.round(2)
})
validation_df.index = ['Level 0 (0例)', 'Level 1 (1-4例)', 'Level 2 (5-49例)', 'Level 3 (>=50例)']
print(validation_df)
'''