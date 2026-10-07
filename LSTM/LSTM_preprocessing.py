import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from config import (CONTINUOUS_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES,
                    USE_VAL,VAL_YEAR,Dengue_dataset)
'''
**********************************************************************
資料前處理
_transform_features :資料處理，categorical => OHE, continuous => SS
preprocessing_final : 切分 train、(val)、test，並用 _split 標記
**********************************************************************
'''

def _transform_features(train_raw, *other_raws):
    """只用 train fit，再轉換任意數量的其他資料切分。"""
    #要做 OHE 的 Feature
    cols_to_encode = list(CATEGORICAL_FEATURES)

    train_town = train_raw['Town'].values
    other_towns = [raw['Town'].values for raw in other_raws]

    if cols_to_encode:
        #有 categorical feature => Do OHE
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
        #無 categorical feature => Don't OHE
        ohe_columns = []
        train_df = train_raw.copy()
        other_dfs = [raw.copy() for raw in other_raws]

    train_df['Town'] = train_town
    for frame, town in zip(other_dfs, other_towns):
        frame['Town'] = town

    continuous_features = CONTINUOUS_FEATURES
    binary_features = BINARY_FEATURES

    #用 train 資料 fit，其餘切分只 transform，避免Data Leakage
    ss = StandardScaler()
    train_df[continuous_features] = ss.fit_transform(train_df[continuous_features])
    for frame in other_dfs:
        frame[continuous_features] = ss.transform(frame[continuous_features])

    all_features = continuous_features + binary_features + ohe_columns

    return (train_df, *other_dfs, all_features)


def preprocessing_final(split_year=2023):
    '''可調整是否使用 Validation '''
    df = Dengue_dataset()
    
    if USE_VAL:
        #使用驗證集
        train_raw = df[df['Year'] < VAL_YEAR].copy()
        val_raw = df[(df['Year'] >= VAL_YEAR) & (df['Year'] < split_year)].copy()
        test_raw = df[df['Year'] >= split_year].copy()
    
        train_df, val_df, test_df, all_features = _transform_features(train_raw, val_raw, test_raw)  
        train_df = train_df.assign(_split='train')
        val_df = val_df.assign(_split='val')
        test_df = test_df.assign(_split='test')
        full_df = pd.concat([train_df, val_df, test_df]).sort_index()
        
    else:    
        #不使用驗證集
        train_raw = df[df['Year'] < split_year].copy()
        test_raw = df[df['Year'] >= split_year].copy()
    
        train_df, test_df, all_features = _transform_features(train_raw, test_raw)
        train_df = train_df.assign(_split='train')
        test_df = test_df.assign(_split='test')
        full_df = pd.concat([train_df, test_df]).sort_index()
        
    split_labels = full_df.pop('_split')
    return full_df, all_features, split_labels
