import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from LSTM_preprocessing import preprocessing

'''
**********************************************************************
回傳值:
train_loader: train DataLoader
val_loader:   validation DataLoader（訓練集最後一年）
test_loader:  test DataLoader
weights: 處理類別不平衡的權重 Tensor (給 Loss function 用)
dim: 模型輸入的特徵維度大小
**********************************************************************
'''
#計算權重
def compute_cb_weights(y_train, num_classes, beta):
    """
    Class-Balanced Loss 權重 (Cui et al., 2019)
    beta 越接近 1，行為越像單純的樣本數反比；
    beta 越小，權重曲線越平滑，稀有類別不會被無限放大
    """
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = np.where(counts > 0, (1.0 - beta) / np.where(effective_num == 0, 1, effective_num), 0.0)
    weights = weights / weights.sum() * num_classes
    return weights


#建立滑動時間窗
def create_time_windows(data, feature_cols, target_col, window_size):
    X, y = [], []
    for _, group in data.groupby('Town'):
        group = group.sort_values(['Year', 'Week'])
        for i in range(len(group) - window_size):
            X.append(group.iloc[i : i + window_size][feature_cols].values)
            y.append(group.iloc[i + window_size][target_col])
    return np.array(X), np.array(y)

#建立 DataLoader
def dengue_dataloader(window_size=4, batch_size=64, split_year=2023, beta=0.999):

    train_df, val_df, test_df, all_features = preprocessing(split_year=split_year)

    #1.建立滑動時間窗
    X_train, y_train = create_time_windows(train_df, all_features, 'RT_level', window_size)
    X_val,   y_val   = create_time_windows(val_df,   all_features, 'RT_level', window_size)
    X_test,  y_test  = create_time_windows(test_df,  all_features, 'RT_level', window_size)

    #(防錯)
    assert len(X_train) > 0, f"訓練集樣本為空，請檢查 split_year={split_year} 設定"
    assert len(X_val)   > 0, f"驗證集樣本為空，val_year={split_year-1} 可能無資料"
    assert len(X_test)  > 0, f"測試集樣本為空，請檢查 split_year={split_year} 設定"

    #2.紀錄特徵維度
    dim = X_train.shape[2]

    #轉換為 PyTorch Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32)  
    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.long)   
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

    #3.計算不平衡的類別權重（Class-Balanced Loss，只用訓練集計算）
    num_classes = 4
    weights = torch.tensor(compute_cb_weights(y_train, num_classes, beta=beta), dtype=torch.float32)

    #4.建立 DataLoader
    #為了讓學習更穩定把訓練集的 train shuffle 設為 True
    #為了模型結果的畫圖與視覺化 test、val shuffle 設為 false
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor) 
    test_dataset  = TensorDataset(X_test_tensor,  y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, weights, dim