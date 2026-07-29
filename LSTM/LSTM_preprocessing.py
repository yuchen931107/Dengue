import pandas as pd
from sklearn.preprocessing import StandardScaler
from get_Dengue import Dengue_dataset
from sklearn.preprocessing import OneHotEncoder
'''
**********************************************************************
回傳值:
train_df = 訓練資料集
val_df   = 驗證資料集
test_df  = 測試資料集
all_features = 所有特徵
**********************************************************************
'''
def preprocessing(split_year=2023):
    #1.取得資料與基礎清理
    df = Dengue_dataset()
    
    #2.切分資料集（依年份，保持時間順序）
    val_year = split_year - 1                               
    train_raw = df[df['Year'] <  val_year].copy()          
    val_raw   = df[df['Year'] == val_year].copy()        
    test_raw  = df[df['Year'] >= split_year].copy()

    #OHE 前先把 Town 存起來，OHE 後貼回，供 create_time_windows 的 groupby 使用
    train_town = train_raw['Town'].values
    val_town   = val_raw['Town'].values                  
    test_town  = test_raw['Town'].values
    
    #3.OHE編碼
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cols_to_encode = ['Town', 'Month']
    
    #用訓練集 fit+transform，驗證集與測試集only transform
    train_ohe_arr = ohe.fit_transform(train_raw[cols_to_encode])
    val_ohe_arr   = ohe.transform(val_raw[cols_to_encode])
    test_ohe_arr  = ohe.transform(test_raw[cols_to_encode])
    
    #自動取得所有編碼後的欄位名稱（例如產生 Town_xxx, Month_xxx）
    ohe_columns = ohe.get_feature_names_out(cols_to_encode)
    
    #轉回 Pandas DataFrame，並補上欄位名稱與原本的索引 (index)
    train_ohe_df = pd.DataFrame(train_ohe_arr, columns=ohe_columns, index=train_raw.index)
    val_ohe_df   = pd.DataFrame(val_ohe_arr, columns=ohe_columns, index=val_raw.index)
    test_ohe_df  = pd.DataFrame(test_ohe_arr, columns=ohe_columns, index=test_raw.index)
    
    #把原本的資料（刪除舊的 Town, Month）與新生成的 OHE DataFrame 合併
    train_df = pd.concat([train_raw.drop(columns=cols_to_encode), train_ohe_df], axis=1)
    val_df   = pd.concat([val_raw.drop(columns=cols_to_encode), val_ohe_df], axis=1)
    test_df  = pd.concat([test_raw.drop(columns=cols_to_encode), test_ohe_df], axis=1)
    
    # =====================================================================
    # Town 欄位僅供 create_time_windows 的 groupby 分組使用，不進入模型特徵
    train_df['Town'] = train_town
    val_df['Town']   = val_town                                         
    test_df['Town']  = test_town
    
    #定義特徵群組
    continuous_features = [
        'Case_Count','RT',
        'Rainfall' , 'AvgTemp', 'TempRange', 
        'AvgHumidity', 'SunshineHours', 'RainfallHours', 'BI', 
        'CI', 'HI', 'LI', 'AI', 'PI', 'Con100HH'
    ]
    town  = [col for col in train_df.columns if col.startswith('Town_')]
    month = [col for col in train_df.columns if col.startswith('Month_')]
    all_features = continuous_features + town + month

    #4. 特徵標準化
    ss = StandardScaler()
    train_df[continuous_features] = ss.fit_transform(train_df[continuous_features])
    val_df[continuous_features]   = ss.transform(val_df[continuous_features]) 
    test_df[continuous_features]  = ss.transform(test_df[continuous_features])

    return train_df, val_df, test_df, all_features
#Test
train_df, val_df, test_df, all_features=preprocessing(split_year=2023)