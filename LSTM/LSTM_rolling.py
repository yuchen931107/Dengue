import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from get_Dengue import Dengue_dataset
from LSTM_loader import build_all_windows, compute_weights
from LSTM_model import DengueLSTM, FocalLoss
from set_seed import set_seed
from config import WINDOWSIZE, HIDDENSIZE, BATCH, GAMMA, LR, NUM_CLASSES, SPLIT_YEAR, PURGE, LEVEL_NAMES

'''
**********************************************************************
Walk-forward（Rolling-origin）交叉驗證

用途：
  之前一直卡住的問題是「2015是唯一有像樣稀有樣本的年份，
  放進val就沒辦法留在train裡教模型」。這支程式解決這個死結：
  每一折的驗證集只挑一年，其餘所有年份（含2015）都留在該折的訓練集裡。
  2015只在「val=2015」那一折(第2折)缺席，其他8折它依然完整在train裡。

  測試集(2023起)從頭到尾不會出現在這支程式的任何一折裡，
  這支程式只負責告訴你「這組設定在9折上平均表現如何」，
  用來挑超參數、比較做法；最終定案後，還是要另外用config.py
  的正式設定（split_year前一年當val）重新訓練一次、拿test評估一次。

  注意：這支程式的OHE/標準化是獨立寫的二份式版本（只有train/val），
  沒有重用LSTM_preprocessing.py的_apply_ohe——因為那支函式設計成
  一定要吃三份（含test），這裡沒有test可以給，sklearn對0筆資料
  的transform會直接報錯。邏輯（train fit、val只transform）跟
  LSTM_preprocessing.py完全一致，只是拿掉了test那一份。
**********************************************************************
'''

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#第1折驗證=2014年（訓練=2011~2013），最後一折驗證=SPLIT_YEAR-1（即2022），測試集(2023起)全程不碰
FOLD_VAL_YEARS = list(range(2014, SPLIT_YEAR))
'''
CONT = ['Case_Count', 'RT', 'Rainfall', 'AvgTemp', 'TempRange', 'AvgHumidity',
        'SunshineHours', 'RainfallHours', 'BI', 'CI', 'HI', 'LI', 'AI', 'PI', 'Con100HH']
BINARY = ['Medicine']
'''
CONT = ['Case_Count', 'RT']


def _apply_ohe_2way(train_raw, val_raw):
    """跟LSTM_preprocessing._apply_ohe邏輯一致的二份式版本：只用train fit，val只transform。"""
    cols_to_encode = ['Town', 'Month']
    train_town = train_raw['Town'].values
    val_town = val_raw['Town'].values

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_ohe_arr = ohe.fit_transform(train_raw[cols_to_encode])
    val_ohe_arr = ohe.transform(val_raw[cols_to_encode])
    ohe_columns = ohe.get_feature_names_out(cols_to_encode)

    train_ohe_df = pd.DataFrame(train_ohe_arr, columns=ohe_columns, index=train_raw.index)
    val_ohe_df = pd.DataFrame(val_ohe_arr, columns=ohe_columns, index=val_raw.index)

    train_df = pd.concat([train_raw.drop(columns=cols_to_encode), train_ohe_df], axis=1)
    val_df = pd.concat([val_raw.drop(columns=cols_to_encode), val_ohe_df], axis=1)
    train_df['Town'] = train_town
    val_df['Town'] = val_town

    ss = StandardScaler()
    train_df[CONT] = ss.fit_transform(train_df[CONT])
    val_df[CONT] = ss.transform(val_df[CONT])

    town = [c for c in train_df.columns if c.startswith('Town_')]
    month = [c for c in train_df.columns if c.startswith('Month_')]
    #all_features = CONT + BINARY + town + month
    all_features = CONT + town + month
    return train_df, val_df, all_features


def make_fold_data(df, val_year, window_size):
    """
    只用 Year < val_year 當train、Year == val_year 當val，
    Year >= val_year+1 之後的年份完全不放進來（避免未來年份意外混進這一折）。
    """
    fold_df = df[df['Year'] <= val_year].copy()
    train_raw = fold_df[fold_df['Year'] < val_year]
    val_raw = fold_df[fold_df['Year'] == val_year]

    train_df, val_df, all_features = _apply_ohe_2way(train_raw, val_raw)
    train_df = train_df.assign(_split='train')
    val_df = val_df.assign(_split='val')
    full_df = pd.concat([train_df, val_df]).sort_index()
    labels = full_df.pop('_split')

    #embargo同樣套用在這一折的train/val邊界上，避免邊界窗口讓val分數虛高
    X, y, target_idx = build_all_windows(full_df, all_features, 'RT_level', window_size,
                                          split_labels=labels, purge=PURGE)
    lw = labels.loc[target_idx].values
    Xtr, ytr = X[lw == 'train'], y[lw == 'train']
    Xva, yva = X[lw == 'val'], y[lw == 'val']
    return Xtr, ytr, Xva, yva, all_features


def train_fold(Xtr, ytr, Xva, yva, dim, seed, epochs=50, patience=7, quiet=True):
    """訓練一折，回傳「val loss最低」那一輪的模型、best_epoch、以及該輪的macro F1與各級F1。"""
    set_seed(seed)
    weights = torch.tensor(compute_weights(ytr, NUM_CLASSES), dtype=torch.float32).to(DEVICE)
    model = DengueLSTM(input_size=dim, hidden_size=HIDDENSIZE, num_layers=2, num_classes=NUM_CLASSES).to(DEVICE)
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

    macro_f1 = f1_score(yva, preds, average='macro', zero_division=0, labels=list(range(NUM_CLASSES)))
    per_class = f1_score(yva, preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    return model, best_epoch, macro_f1, per_class


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
        _, best_epoch, macro_f1, per_class = train_fold(Xtr, ytr, Xva, yva, dim, seed, epochs, patience)

        rows.append({
            'val_year': val_year,
            'train_years': f"2011-{val_year-1}",
            'n_train': len(Xtr),
            'n_val': len(Xva),
            'best_epoch': best_epoch,
            'macro_f1': macro_f1,
            'per_class': np.round(per_class, 3).tolist(),
        })
        elapsed = time.time() - t0
        print(f"[第{len(rows)}折] val={val_year} (train=2011-{val_year-1}) | "
              f"n_train={len(Xtr):,} n_val={len(Xva):,} | 停在第{best_epoch}輪 | "
              f"macro F1={macro_f1:.4f} | 各級F1={np.round(per_class,3).tolist()} | {elapsed:.0f}秒")

    result_df = pd.DataFrame(rows)
    return result_df


def summarise(result_df):
    macro = result_df['macro_f1'].to_numpy()
    per_class_matrix = np.stack(result_df['per_class'].to_list())   #shape: (折數, NUM_CLASSES)

    print("\n" + "=" * 60)
    print(f"=== {len(result_df)} 折平均結果 ===")
    print(f"Macro F1： {macro.mean():.4f} ± {macro.std(ddof=1):.4f}")
    for i, name in enumerate(LEVEL_NAMES):
        col = per_class_matrix[:, i]
        print(f"{name} F1： {col.mean():.4f} ± {col.std(ddof=1):.4f}")
    print("=" * 60)
    print("\n※ 測試集(2023起)全程未使用，以上只是內部驗證結果，")
    print("   拿來挑超參數/比較做法；最終定案後仍須另外用完整train重新訓練、跑一次test。")


if __name__ == '__main__':
    result_df = run_rolling_cv(seed=1234, epochs=50, patience=7)
    print("\n=== 各折明細 ===")
    print(result_df[['val_year', 'train_years', 'n_train', 'n_val', 'best_epoch', 'macro_f1']]
          .to_string(index=False))
    summarise(result_df)