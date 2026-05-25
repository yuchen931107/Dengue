import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from Dengue_level import level
from get_Dengue import Dengue_dataset
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
'''
**********************************************************************
回傳值:
train_loader: 訓練集 DataLoader
test_loader: 測試集 DataLoader
class_weights_tensor: 處理類別不平衡的權重 Tensor (給 Loss function 用)
input_dim: 模型輸入的特徵維度大小
**********************************************************************
此處進行 LSTM 的資料預處理，主要流程包含：
1. 資料載入與清理：載入原始登革熱數據，產生預測目標 ，移除不必要的滯後 (Lag) 特徵
2. 類別變數轉換：針對鄉鎮與月份進行 One-Hot Encoding
3. 資料集切分：依據年份切分訓練集與測試集，避免時間序列預測產生未來資料洩漏
4. 特徵縮放：依特徵屬性分別使用 StandardScaler 與 MinMaxScaler
5. 建立滑動時間窗 ：將 2D表格資料轉換為 LSTM 所需的 3D 張量格式 (Samples, Time Steps, Features)。
6. Tensor 轉換：將特徵與標籤轉換為 PyTorch 支援的 Tensor 格式 。
7. 類別不平衡處理：計算各等級的 balanced 權重。
8. 建立 DataLoader：將資料封裝並打包，訓練集啟用 shuffle，測試集關閉 shuffle(保持時間順序)。
**********************************************************************
Note:
極端值特徵 (StandardScaler)：Rainfall（颱風天會有暴雨極端值）、Population density（市區與偏鄉落差極大）。
常規範圍特徵 (MinMaxScaler)：氣溫、濕度、時數與各類蟲媒指數（通常有合理的上下限）。
**********************************************************************
'''
#先依鄉鎮各自建窗再合併，時間序列才不會搞混
def create_time_windows(data, feature_cols, target_col, window_size):
    X, y = [], []
    for _, group in data.groupby('Town'):                   
        group = group.sort_values(['Year', 'Week'])                   
        for i in range(len(group) - window_size):
            X.append(group.iloc[i : i + window_size][feature_cols].values)
            y.append(group.iloc[i + window_size][target_col])
    return np.array(X), np.array(y)

def dengue_dataloader(window_size=4, batch_size=64, split_year=2024):

    #1.取得資料與基礎清理
    df = Dengue_dataset()
    df = level(df, case_column='Case_Count', new_column='Dengue_level')
    df = df.drop(columns=['Case_Lag1W','Case_Lag2W','Case_Lag3W',
                          'Temp_Lag2W','Temp_Lag4W','Temp_Lag6W',
                          'Temp_Lag8W','Rain_Lag2W','Rain_Lag4W',
                          'Rain_Lag6W','Rain_Lag8W','BI_Lag2W',
                          'CI_Lag2W','HI_Lag2W','LI_Lag2W','AI_Lag2W',
                          'PI_Lag2W','Con100HH_Lag2W'])
    
    #2.切分資料集
    train_raw = df[df['Year'] < split_year].copy()
    test_raw  = df[df['Year'] >= split_year].copy()
    
    #3. 處理類別變數(One-Hot Encoding)
    train_town = train_raw['Town'].values  
    test_town  = test_raw['Town'].values   
    
    train_df = pd.get_dummies(train_raw, columns=['Town', 'Month'], dtype=float)
    test_df  = pd.get_dummies(test_raw,  columns=['Town', 'Month'], dtype=float)
    
    train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)
    train_df['Town'] = train_town       
    test_df['Town']  = test_town     
    
    #定義特徵群組
    standard = ['Rainfall', 'Population density']
    minmax = ['AvgTemp', 'TempRange', 'AvgHumidity', 'SunshineHours', 
              'RainfallHours', 'BI', 'CI', 'HI', 'LI','AI','PI',
              'Con100HH','RT'
    ]
    town  = [col for col in train_df.columns if col.startswith('Town_')]
    month = [col for col in train_df.columns if col.startswith('Month_')]
    all_features = standard + minmax + town + month

    #4.數值特徵縮放
    scaler_std = StandardScaler()
    scaler_mm = MinMaxScaler()

    train_df[standard] = scaler_std.fit_transform(train_df[standard])
    train_df[minmax] = scaler_mm.fit_transform(train_df[minmax])
    test_df[standard] = scaler_std.transform(test_df[standard])
    test_df[minmax] = scaler_mm.transform(test_df[minmax])

    #5.建立滑動時間窗
    X_train, y_train = create_time_windows(train_df, all_features, 'Dengue_level', window_size)
    X_test ,  y_test  = create_time_windows(test_df, all_features, 'Dengue_level', window_size)
    
    #提前檢查測試集是否為空，避免靜默失敗
    assert len(X_train) > 0, f"訓練集樣本為空，請檢查 split_year={split_year} 設定"
    assert len(X_test)  > 0, f"測試集樣本為空，請檢查 split_year={split_year} 設定"

    #紀錄特徵維度
    input_dim = X_train.shape[2] 

    #6.轉換為 PyTorch Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    #7.計算對抗不平衡的類別權重
    num_classes = 4  # Level 0/1/2/3
    unique_classes  = np.unique(y_train)
    actual_weights  = compute_class_weight(class_weight='balanced',
                                           classes=unique_classes,
                                           y=y_train)
    weight_dict     = dict(zip(unique_classes, actual_weights))
    final_weights   = [weight_dict.get(i, 1.0) for i in range(num_classes)]   # ← 修改
    class_weights_tensor = torch.tensor(final_weights, dtype=torch.float32)

    #8.建立 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, class_weights_tensor, input_dim