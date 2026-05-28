import pandas as pd
import numpy as np

def RT_split(df, rt_column='RT', new_column='RT_level'):
    """
    將連續 RT切分為 4 個預警等級：
    Level 0: 0 
    Level 1: >0&&<=1
    Level 2: >1&&<=2
    Level 3: >2
    """
    # 切分點改為：負無限大到0, 0到1, 1到2 , 2到無限大
    bins = [-np.inf, 0, 1, 2 ,np.inf]
    labels = [0, 1, 2 , 3] 
    
    df[new_column] = pd.cut(df[rt_column], bins=bins, labels=labels)
    return df
