import pandas as pd
from sklearn.preprocessing import StandardScaler
from get_Dengue import Dengue_dataset
from sklearn.preprocessing import OneHotEncoder
from config import CONTINUOUS_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES
'''
**********************************************************************
本檔案提供兩種前處理進入點：

preprocessing(split_year, val_ranges)
    舊版行為：分別把 train/val/test 切成三份獨立的 DataFrame，
    OHE 只在各自的 DataFrame 內處理。適合單純想拿到「切好的三份資料」
    做簡單分析（例如 feature importance 只需要 test_df）的情境。
    缺點：若切分邊界前後不連續（例如自訂 val_ranges 挑出中段的爆發區間），
    配合 create_time_windows 使用時，切分邊界附近會因為缺少足夠的
    「前情提要」週數而被迫捨棄，可能誤傷關鍵樣本。

preprocessing_full(split_year, val_ranges)
    新版行為：回傳「整個」時間軸的單一 DataFrame（含 OHE 特徵），
    以及每一列屬於 train/val/test 哪個切分的標籤（split_labels）。
    真正的 train/val/test 窗口是在 LSTM_loader.build_all_windows 裡，
    對整個時間軸建完所有合法窗口後，才依照每個窗口「target 那一週」
    的日期去分配歸屬——這樣切分邊界附近的資料就不會被誤傷。
    dengue_dataloader 用的是這一版。

preprocessing_final(split_year)
    正式 fixed-epoch 訓練用：2011 至 split_year 前一年全部歸入 train，
    不保留 validation；test 仍從 split_year 開始，且只以完整 train fit 前處理器。
**********************************************************************
'''

def _split_masks(df, split_year, val_ranges):
    """依照 split_year / val_ranges，回傳 train/val/test 三個布林遮罩（皆以 df 的 index 對齊）"""
    week_dt = pd.to_datetime(df['Week'])
    test_mask = df['Year'] >= split_year

    if val_ranges is None:
        val_year = split_year - 1
        val_mask = (~test_mask) & (df['Year'] == val_year)
    else:
        val_mask = pd.Series(False, index=df.index)
        for start, end in val_ranges:
            val_mask |= (week_dt >= pd.Timestamp(start)) & (week_dt <= pd.Timestamp(end))
        val_mask = val_mask & (~test_mask)

    train_mask = (~test_mask) & (~val_mask)
    return train_mask, val_mask, test_mask


def _transform_features(train_raw, *other_raws):
    """只用 train fit 前處理器，再轉換任意數量的其他資料切分。"""
    #哪些欄位要One-Hot：統一從config.py讀（CATEGORICAL_FEATURES），不在這裡寫死
    cols_to_encode = list(CATEGORICAL_FEATURES)

    #Town不管有沒有被當成特徵，建滑動窗口時都要用它分組，所以先存起來、最後補回
    train_town = train_raw['Town'].values
    other_towns = [raw['Town'].values for raw in other_raws]

    if cols_to_encode:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        train_ohe_arr = ohe.fit_transform(train_raw[cols_to_encode])
        ohe_columns = list(ohe.get_feature_names_out(cols_to_encode))

        train_ohe_df = pd.DataFrame(train_ohe_arr, columns=ohe_columns, index=train_raw.index)
        train_df = pd.concat([train_raw.drop(columns=cols_to_encode), train_ohe_df], axis=1)
        other_dfs = []
        for raw in other_raws:
            raw_ohe_arr = ohe.transform(raw[cols_to_encode])
            raw_ohe_df = pd.DataFrame(raw_ohe_arr, columns=ohe_columns, index=raw.index)
            other_dfs.append(pd.concat([raw.drop(columns=cols_to_encode), raw_ohe_df], axis=1))
    else:
        #這個特徵集不使用任何類別型特徵，就不做One-Hot（OneHotEncoder不能fit在0個欄位上）
        ohe_columns = []
        train_df = train_raw.copy()
        other_dfs = [raw.copy() for raw in other_raws]

    train_df['Town'] = train_town
    for frame, town in zip(other_dfs, other_towns):
        frame['Town'] = town

    #連續型、二元型特徵：統一從config.py讀，切FEATURE_SET就能同步切換，
    #不用再跑來這裡跟LSTM_rolling.py各自手動改一次、容易忘記同步
    continuous_features = CONTINUOUS_FEATURES
    binary_features = BINARY_FEATURES

    #用 train 資料 fit，其餘切分只 transform，避免用到未來資訊
    ss = StandardScaler()
    train_df[continuous_features] = ss.fit_transform(train_df[continuous_features])
    for frame in other_dfs:
        frame[continuous_features] = ss.transform(frame[continuous_features])

    #One-Hot展開後的欄位順序 = cols_to_encode的順序（例如Town_*全部，接著Month_*全部）
    all_features = continuous_features + binary_features + ohe_columns

    return (train_df, *other_dfs, all_features)


def _apply_ohe(train_raw, val_raw, test_raw):
    """相容既有 train/val/test 進入點的三份式前處理包裝。"""
    return _transform_features(train_raw, val_raw, test_raw)


#舊版：回傳三份「各自獨立」的 DataFrame（配合 create_time_windows 逐一使用）
def preprocessing(split_year=2023, val_ranges=None):
    df = Dengue_dataset()
    train_mask, val_mask, test_mask = _split_masks(df, split_year, val_ranges)

    train_raw = df[train_mask].copy()
    val_raw   = df[val_mask].copy()
    test_raw  = df[test_mask].copy()

    train_df, val_df, test_df, all_features = _apply_ohe(train_raw, val_raw, test_raw)
    return train_df, val_df, test_df, all_features


#新版：回傳整個時間軸單一 DataFrame + 每列的切分標籤（配合 build_all_windows 使用）
def preprocessing_full(split_year=2023, val_ranges=None):
    df = Dengue_dataset()
    train_mask, val_mask, test_mask = _split_masks(df, split_year, val_ranges)

    train_raw = df[train_mask].copy()
    val_raw   = df[val_mask].copy()
    test_raw  = df[test_mask].copy()

    train_df, val_df, test_df, all_features = _apply_ohe(train_raw, val_raw, test_raw)

    #把三份切分好的資料重新合併回「整個時間軸」，並記錄每一列屬於哪個切分
    train_df = train_df.assign(_split='train')
    val_df   = val_df.assign(_split='val')
    test_df  = test_df.assign(_split='test')

    full_df = pd.concat([train_df, val_df, test_df]).sort_index()
    split_labels = full_df.pop('_split')

    return full_df, all_features, split_labels


def preprocessing_final(split_year=2023):
    """正式 fixed-epoch 訓練：以 split_year 前的全部資料 fit，保留 split_year 起的 test。"""
    df = Dengue_dataset()
    train_raw = df[df['Year'] < split_year].copy()
    test_raw = df[df['Year'] >= split_year].copy()

    train_df, test_df, all_features = _transform_features(train_raw, test_raw)
    train_df = train_df.assign(_split='train')
    test_df = test_df.assign(_split='test')
    full_df = pd.concat([train_df, test_df]).sort_index()
    split_labels = full_df.pop('_split')

    return full_df, all_features, split_labels


#Test
if __name__ == '__main__':
    train_df, val_df, test_df, all_features = preprocessing(split_year=2023)
    print(f"[preprocessing] train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    full_df, all_features, split_labels = preprocessing_full(split_year=2023)
    print(f"[preprocessing_full] total: {len(full_df)}  labels: {split_labels.value_counts().to_dict()}")
