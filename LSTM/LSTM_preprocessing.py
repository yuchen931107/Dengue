import pandas as pd
from sklearn.preprocessing import StandardScaler
from get_Dengue import Dengue_dataset
from sklearn.preprocessing import OneHotEncoder
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


def _apply_ohe(train_raw, val_raw, test_raw):
    """用 train 資料 fit OHE，三份資料各自 transform，回傳合併好 OHE 欄位、且補回 Town 欄位的 df"""
    cols_to_encode = ['Town', 'Month']

    train_town = train_raw['Town'].values
    val_town   = val_raw['Town'].values
    test_town  = test_raw['Town'].values

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_ohe_arr = ohe.fit_transform(train_raw[cols_to_encode])
    val_ohe_arr   = ohe.transform(val_raw[cols_to_encode])
    test_ohe_arr  = ohe.transform(test_raw[cols_to_encode])

    ohe_columns = ohe.get_feature_names_out(cols_to_encode)

    train_ohe_df = pd.DataFrame(train_ohe_arr, columns=ohe_columns, index=train_raw.index)
    val_ohe_df   = pd.DataFrame(val_ohe_arr, columns=ohe_columns, index=val_raw.index)
    test_ohe_df  = pd.DataFrame(test_ohe_arr, columns=ohe_columns, index=test_raw.index)

    train_df = pd.concat([train_raw.drop(columns=cols_to_encode), train_ohe_df], axis=1)
    val_df   = pd.concat([val_raw.drop(columns=cols_to_encode), val_ohe_df], axis=1)
    test_df  = pd.concat([test_raw.drop(columns=cols_to_encode), test_ohe_df], axis=1)

    train_df['Town'] = train_town
    val_df['Town']   = val_town
    test_df['Town']  = test_town

    continuous_features = [
        'Case_Count', 'RT',
        'Rainfall', 'AvgTemp', 'TempRange',
        'AvgHumidity', 'SunshineHours', 'RainfallHours', 'BI',
        'CI', 'HI', 'LI', 'AI', 'PI', 'Con100HH'
    ]
    town  = [col for col in train_df.columns if col.startswith('Town_')]
    month = [col for col in train_df.columns if col.startswith('Month_')]
    all_features = continuous_features + town + month

    return train_df, val_df, test_df, all_features


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


#Test
if __name__ == '__main__':
    train_df, val_df, test_df, all_features = preprocessing(split_year=2023)
    print(f"[preprocessing] train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    full_df, all_features, split_labels = preprocessing_full(split_year=2023)
    print(f"[preprocessing_full] total: {len(full_df)}  labels: {split_labels.value_counts().to_dict()}")