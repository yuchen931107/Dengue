import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from LSTM_preprocessing import preprocessing_final
from config import NUM_CLASSES, PURGE, WINDOWSIZE, BATCH, SPLIT_YEAR

'''
**********************************************************************
資料打包
compute_weights   : 計算 Focal loss 的 alpha
build_all_windows : 打包成 X+y 的 型態
dengue_dataloader : 切分資料集、計算權重並封裝 batch 成各 DataLoader
**********************************************************************
'''

'''
計算 Focal Loss 使用的類別權重 (Alpha)，處理資料極端不平衡問題：
1. 樣本統計：清點訓練集中各疫情等級的實際數量。
2. 反比平衡：套用公式 Total / (Num_Classes * Class_Count) 確保各類別對總 Loss 的理論貢獻度相等，數量越少的類別權重越高。
3. 平滑化處理：將上述平衡權重開平方根 sqrt(Balanced)，防止極度稀少類別（如 Level 2）獲得暴衝的權重，避免模型過度反應與誤報。
'''
def compute_weights(y_train, num_classes):
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    balanced = total / (num_classes * counts)
    weights = np.sqrt(balanced)
    return weights


#對整個資料集建立所有時間連續的滑動窗口
def build_all_windows(data, feature_cols, target_col, window_size, split_labels=None, purge=False):
    X, y, target_index = [], [], []
    data = data.copy()
    data['_week_dt'] = pd.to_datetime(data['Week'])
    data = data.reset_index().rename(columns={'index': '_orig_idx'})

    skipped_gap = 0      #時間斷層計數
    skipped_purge = {}   #embargo

    #於同一區的資料開始滑動建置窗口
    for _, group in data.groupby('Town'):
        group = group.sort_values('_week_dt').reset_index(drop=True)
        for i in range(len(group) - window_size):
            span = group.iloc[i : i + window_size + 1]
            week_diffs = span['_week_dt'].diff().dropna()
            #窗口(含target)內任何一段間隔不是剛好7天，代表跨越了不連續的時間斷層
            if not (week_diffs == pd.Timedelta(days=7)).all():
                skipped_gap += 1
                continue

            #檢查窗口的「輸入週」跟「target週」是不是都屬於同一個切分。
            if purge and split_labels is not None:
                span_splits = split_labels.loc[span['_orig_idx'].values].values
                if len(set(span_splits)) > 1:
                    target_split = span_splits[-1]
                    skipped_purge[target_split] = skipped_purge.get(target_split, 0) + 1
                    continue

            X.append(group.iloc[i : i + window_size][feature_cols].values)
            y.append(group.iloc[i + window_size][target_col])
            target_index.append(group.iloc[i + window_size]['_orig_idx'])

    if skipped_gap > 0:
        print(f"[build_all_windows] 偵測到時間斷層，已跳過 {skipped_gap} 個不連續窗口")
    if skipped_purge:
        for sp in ('train', 'val', 'test'):
            if sp in skipped_purge:
                print(f"[embargo] {sp}：丟掉 {skipped_purge[sp]} 個輸入跨切分的窗口")

    return np.array(X), np.array(y), np.array(target_index)


#建立 DataLoader
def dengue_dataloader(window_size=WINDOWSIZE, batch_size=BATCH, split_year=SPLIT_YEAR, purge=PURGE):
    full_df, all_features, split_labels = preprocessing_final(split_year=split_year)

    #對整個資料集一次建好所有合法窗口
    X_all, y_all, target_idx = build_all_windows(full_df, all_features, 'RT_level', window_size,
                                                  split_labels=split_labels, purge=purge)

    #依照每個窗口 target 那一列原本被標記的切分，分配窗口歸屬
    labels_for_windows = split_labels.loc[target_idx].values

    train_mask = labels_for_windows == 'train'
    test_mask  = labels_for_windows == 'test'
    val_mask   = labels_for_windows == 'val'

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    #(防錯)
    assert len(X_train) > 0, f"訓練集樣本為空，請檢查 split_year={split_year} 設定"
    assert len(X_test)  > 0, f"測試集樣本為空，請檢查 split_year={split_year} 設定"

    dim = X_train.shape[2]

    #轉換成 PyTorch 可讀取運算的結構
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

    weights = torch.tensor(compute_weights(y_train, NUM_CLASSES), dtype=torch.float32)

    #合併 X、y
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset  = TensorDataset(X_test_tensor,  y_test_tensor)

    #打包成 batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    #如果有 val 資料，建立 val_loader
    val_loader = None
    if val_mask.any():
        X_val, y_val = X_all[val_mask], y_all[val_mask]
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, weights, dim

