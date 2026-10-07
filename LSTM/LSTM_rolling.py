import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from LSTM_preprocessing import _transform_features
from LSTM_loader import build_all_windows, compute_weights
from LSTM_model import DengueLSTM, FocalLoss
from set_seed import set_seed
from config import (WINDOWSIZE, HIDDENSIZE, BATCH, GAMMA, LR, NUM_CLASSES, SPLIT_YEAR, PURGE, 
                    LEVEL_NAMES, CONTINUOUS_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, SEED,
                    NUM_LAYERS, MAX_EPOCH, PATIENCE,Dengue_dataset)

'''
**********************************************************************
Walk-forward 滾動式交叉驗證
用途：
每一折的驗證集只挑一年，其餘驗證集之前的所有年份(val除外)都留在該折的訓練集裡。
這支程式用來挑超參數、比較做法
**********************************************************************
'''

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#第1折驗證=2014年（訓練=2011~2013），最後一折驗證=SPLIT_YEAR-1（即2022），測試集(2023起)全程不碰
FOLD_VAL_YEARS = list(range(2014, SPLIT_YEAR))
CONT = CONTINUOUS_FEATURES
BINARY = BINARY_FEATURES
CATEGORICAL = CATEGORICAL_FEATURES

def make_fold_data(df, val_year, window_size):
    fold_df = df[df['Year'] <= val_year].copy()
    train_raw = fold_df[fold_df['Year'] < val_year]
    val_raw = fold_df[fold_df['Year'] == val_year]

    train_df, val_df, all_features = _transform_features(train_raw, val_raw)
    train_df = train_df.assign(_split='train')
    val_df = val_df.assign(_split='val')
    full_df = pd.concat([train_df, val_df]).sort_index()
    labels = full_df.pop('_split')

    #embargo同樣套用在 train/val 邊界上
    X, y, target_idx = build_all_windows(full_df, all_features, 'RT_level', window_size,
                                          split_labels=labels, purge=PURGE)
    lw = labels.loc[target_idx].values
    Xtr, ytr = X[lw == 'train'], y[lw == 'train']
    Xva, yva = X[lw == 'val'], y[lw == 'val']
    return Xtr, ytr, Xva, yva, all_features


def train_fold(Xtr, ytr, Xva, yva, dim, seed, epochs=MAX_EPOCH, patience=PATIENCE, quiet=True):
    """訓練一折，回傳「val loss最低」那一輪的模型、best_epoch、以及該輪的macro F1與各級F1。"""
    set_seed(seed)
    weights = torch.tensor(compute_weights(ytr, NUM_CLASSES), dtype=torch.float32).to(DEVICE)
    model = DengueLSTM(input_size=dim, hidden_size=HIDDENSIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = FocalLoss(alpha=weights, gamma=GAMMA)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)),
        batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.long)),
        batch_size=BATCH, shuffle=False)

    best_val_loss, best_state, best_epoch, stall = float('inf'), None, 0, 0

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                val_loss += criterion(model(bx.to(DEVICE)), by.to(DEVICE)).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, stall = val_loss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= patience:
                break
        if not quiet:
            print(f"    epoch {epoch+1:2d} | val loss {val_loss:.4f}")

    model.load_state_dict(best_state)

    #用best_epoch那輪的權重，算這一折的val表現
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in val_loader:
            preds.append(model(bx.to(DEVICE)).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)

    #只平均「這一折實際存在」的類別
    present = sorted(set(yva.tolist()))
    macro_f1 = f1_score(yva, preds, average='macro', zero_division=0, labels=present)
    per_class = f1_score(yva, preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    n_pos = np.bincount(yva, minlength=NUM_CLASSES)
    
    return model, best_epoch, macro_f1, per_class, n_pos


def run_rolling_cv(seed=1234, epochs=50, patience=7):
    print("=== 載入資料 ===")
    df = Dengue_dataset()

    rows = []
    for val_year in FOLD_VAL_YEARS:
        t0 = time.time()
        Xtr, ytr, Xva, yva, _ = make_fold_data(df, val_year, WINDOWSIZE)

        if len(Xva) == 0:
            print(f"[跳過] val={val_year}：這一折驗證集窗口數為0（可能被embargo全部濾掉），無法評估")
            continue

        dim = Xtr.shape[2]
        #接收新增的 n_pos
        _, best_epoch, macro_f1, per_class, n_pos = train_fold(Xtr, ytr, Xva, yva, dim, seed, epochs, patience)

        rows.append({
            'val_year': val_year,
            'train_years': f"2011-{val_year-1}",
            'n_train': len(Xtr),
            'n_val': len(Xva),
            'best_epoch': best_epoch,
            'macro_f1': macro_f1,
            'per_class': np.round(per_class, 3).tolist(),
            'n_pos': n_pos.tolist(),
        })
        elapsed = time.time() - t0
        print(f"[第{len(rows)}折] val={val_year} (train=2011-{val_year-1}) | "
              f"n_train={len(Xtr):,} n_val={len(Xva):,} | 停在第{best_epoch}輪 | "
              f"macro F1={macro_f1:.4f} | 各級F1={np.round(per_class,3).tolist()} | {elapsed:.0f}秒")

    result_df = pd.DataFrame(rows)
    return result_df


def summarise(result_df):
    macro = result_df['macro_f1'].to_numpy()
    per_class_matrix = np.stack(result_df['per_class'].to_list())   
    n_pos_matrix = np.stack(result_df['n_pos'].to_list())           

    print("\n" + "=" * 60)
    print(f"=== {len(result_df)} 折平均結果 ===")
    print(f"Macro F1： {macro.mean():.4f} ± {macro.std(ddof=1):.4f}")
    
    for i, name in enumerate(LEVEL_NAMES):
        #只挑出「該折驗證集確實有這個類別 (n_pos > 0)」的成績來平均
        valid_folds_mask = n_pos_matrix[:, i] > 0
        if valid_folds_mask.sum() > 0:
            col_valid = per_class_matrix[valid_folds_mask, i]
            print(f"{name} F1： {col_valid.mean():.4f} ± {col_valid.std(ddof=1):.4f} (基於 {valid_folds_mask.sum()} 折有效驗證)")
        else:
            print(f"{name} F1： 無效 (所有折均無此類別)")
            
    print("=" * 60)
    print("\n※ 測試集(2023起)全程未使用，以上只是內部驗證結果，")
    print("   拿來挑超參數/比較做法；最終定案後仍須另外用完整train重新訓練、跑一次test。")

if __name__ == '__main__':
    result_df = run_rolling_cv(seed=SEED, epochs=MAX_EPOCH, patience=PATIENCE)

    print("\n=== 各折明細 ===")
    print(result_df[['val_year', 'train_years', 'n_train', 'n_val', 'best_epoch', 'macro_f1']]
          .to_string(index=False))
    summarise(result_df)
