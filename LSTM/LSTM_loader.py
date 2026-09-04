import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from LSTM_preprocessing import preprocessing_full

'''
**********************************************************************
回傳值:
train_loader: train DataLoader
val_loader:   validation DataLoader
test_loader:  test DataLoader
weights: 處理類別不平衡的權重 Tensor (給 Loss function 用)
dim: 模型輸入的特徵維度大小

設計原則（重要）：
先對「整個」時序（不分 train/val/test）建立所有時間連續的滑動窗口，
每個窗口只依「target 那一週」落在哪個日期範圍，分配到 train/val/test。
這樣即使某個窗口的「特徵歷史」往前借用了另一個切分的資料，也沒關係，
因為那只是拿過去已發生的天氣/病媒資料當輸入，不會用到任何其他切分
自己的 label——這是時間序列預測的標準做法，也是為了避免「切分邊界
附近的資料因為缺少前情提要而被迫捨棄」這個問題（尤其是像 2015 下半年
這種刻意挑選的爆發段，開頭幾週最容易被誤傷）。
**********************************************************************
'''
#計算權重
def compute_cb_weights(y_train, num_classes, beta):
    """
    Class-Balanced Loss 權重 (Cui et al., 2019)
    """
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = np.where(counts > 0, (1.0 - beta) / np.where(effective_num == 0, 1, effective_num), 0.0)
    weights = weights / weights.sum() * num_classes
    return weights


#(舊版/單一 DataFrame 用) 對單一切分好的 DataFrame 建立滑動窗口，含時間連續性檢查
#適用情境：資料本身就是連續時間軸的單一區塊（例如 feature importance 只需要 test_df）
#若要切出中段不連續的 val（如 2015下半年+2022），請改用下面的 build_all_windows + preprocessing_full
def create_time_windows(data, feature_cols, target_col, window_size):
    X, y = [], []
    data = data.copy()
    data['_week_dt'] = pd.to_datetime(data['Week'])

    skipped = 0
    for _, group in data.groupby('Town'):
        group = group.sort_values('_week_dt').reset_index(drop=True)
        for i in range(len(group) - window_size):
            span = group.iloc[i : i + window_size + 1]
            week_diffs = span['_week_dt'].diff().dropna()
            if not (week_diffs == pd.Timedelta(days=7)).all():
                skipped += 1
                continue
            X.append(group.iloc[i : i + window_size][feature_cols].values)
            y.append(group.iloc[i + window_size][target_col])

    if skipped > 0:
        print(f"[create_time_windows] 偵測到時間斷層，已跳過 {skipped} 個不連續窗口")

    return np.array(X), np.array(y)


#對「整個」資料集建立所有時間連續的滑動窗口，並記錄每個窗口 target 那一列的原始 index
def build_all_windows(data, feature_cols, target_col, window_size):
    X, y, target_index = [], [], []
    data = data.copy()
    data['_week_dt'] = pd.to_datetime(data['Week'])
    data = data.reset_index().rename(columns={'index': '_orig_idx'})

    skipped = 0
    for _, group in data.groupby('Town'):
        group = group.sort_values('_week_dt').reset_index(drop=True)
        for i in range(len(group) - window_size):
            span = group.iloc[i : i + window_size + 1]
            week_diffs = span['_week_dt'].diff().dropna()
            #窗口(含target)內任何一段間隔不是剛好7天，代表跨越了不連續的時間斷層
            if not (week_diffs == pd.Timedelta(days=7)).all():
                skipped += 1
                continue
            X.append(group.iloc[i : i + window_size][feature_cols].values)
            y.append(group.iloc[i + window_size][target_col])
            target_index.append(group.iloc[i + window_size]['_orig_idx'])

    if skipped > 0:
        print(f"[build_all_windows] 偵測到時間斷層，已跳過 {skipped} 個不連續窗口")

    return np.array(X), np.array(y), np.array(target_index)


#建立 DataLoader
def dengue_dataloader(window_size=4, batch_size=64, split_year=2023, beta=0.999, val_ranges=None):
    """
    val_ranges: 若為 None，val = split_year 前一整年（與舊版行為一致）。
                若要自訂驗證集區間，傳入 [(start, end), ...] 日期字串 list，
                例如 [('2015-09-01','2015-12-31'), ('2022-01-01','2022-12-31')]。
                「窗口的特徵歷史」可以跨越切分邊界往前借用資料，
                但「窗口是哪個切分」永遠只看 target 那一週的日期。
    """
    full_df, all_features, split_labels = preprocessing_full(split_year=split_year, val_ranges=val_ranges)

    #對整個資料集一次建好所有合法窗口
    X_all, y_all, target_idx = build_all_windows(full_df, all_features, 'RT_level', window_size)

    #依照每個窗口 target 那一列原本被標記的切分（train/val/test），分配窗口歸屬
    labels_for_windows = split_labels.loc[target_idx].values

    train_mask = labels_for_windows == 'train'
    val_mask   = labels_for_windows == 'val'
    test_mask  = labels_for_windows == 'test'

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    #(防錯)
    assert len(X_train) > 0, f"訓練集樣本為空，請檢查 split_year={split_year} 設定"
    assert len(X_val)   > 0, f"驗證集樣本為空，請檢查 val_ranges/{split_year} 設定"
    assert len(X_test)  > 0, f"測試集樣本為空，請檢查 split_year={split_year} 設定"

    dim = X_train.shape[2]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

    num_classes = 4
    weights = torch.tensor(compute_cb_weights(y_train, num_classes, beta=beta), dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)
    test_dataset  = TensorDataset(X_test_tensor,  y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, weights, dim