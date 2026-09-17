import pandas as pd
import numpy as np

def RT_split(df, rt_column='RT', new_column='RT_level'):
    """
    將連續 RT 切分為 3 個預警等級：
    Level 0：RT <= 0.01        → 無有效訊號
    Level 1：0.01 < RT <= 1.0  → 有真實訊號，低於流行病學閾值
    Level 2：RT > 1.0          → 高於閾值，疫情成長中
    """
    # 切分點改為：負無限大到0.01, 0.01到1.0, 1.0到無限大
    bins = [-np.inf, 0.01, 1.0, np.inf]
    labels = [0, 1, 2] 
    
    # 使用 right=True，讓區間變成 (a, b]，也就是包含右邊界
    # 這樣剛好符合 RT <= 0.01 與 RT <= 1.0 的邏輯
    df[new_column] = pd.cut(df[rt_column], bins=bins, labels=labels, right=True)
    return df