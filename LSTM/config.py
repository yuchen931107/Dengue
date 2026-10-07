'''
**********************************************************************
共用設定檔：所有腳本共用的參數，
**********************************************************************
'''
#抓取資料集
import pandas as pd
def Dengue_dataset():
    url = "https://raw.githubusercontent.com/yuchen931107/Dengue/refs/heads/main/data/Tainan_Dengue_ML.csv"
    df = pd.read_csv(url)
    return df

#參數調整
SEED = 1234         #隨機種子碼
WINDOWSIZE = 4      #滑動時間窗大小（週數）
HIDDENSIZE = 64     #LSTM 隱藏層大小，影響模型結構
BATCH = 64          #訓練用的batch size
GAMMA = 2           #Focal Loss 的 gamma 參數
DROPOUT = 0.3       #神經元隨機關閉的趴數、防止Overfitting
LR = 0.001          #Adam optimizer的學習率
NUM_LAYERS = 2      #神經網路層數

#訓練控制參數
PURGE = True        #embargo：丟掉輸入週跨越 train/test 邊界的窗口。
USE_VAL = False     #是否使用驗證集 IF 使用會使用 Early stopped Else 使用 Fixed Epoch
VAL_YEAR = 2022     #驗證集年份
PATIENCE = 7        #Early stopped 值
MAX_EPOCH = 50      #最大輪數
FIXED_EPOCHS = 20   #訓練時固定輪數



#Target 分級
NUM_CLASSES = 3
LEVEL_NAMES = ['Level 0', 'Level 1', 'Level 2']

SPLIT_YEAR = 2023             #Test set = (Year >= SPLIT_YEAR)
TESTYEAR_LABEL = "2023-2025"  #圖表標題
'''
==========================================================================
特徵設定：所有「用哪些特徵建模」的決定都集中在這裡，
三種特徵型態，處理方式不同：
continuous  ：連續型，會做標準化（StandardScaler，只用train fit）
binary      ：本身就是0/1，不標準化，直接輸入模型
categorical ：類別型，會做One-Hot編碼，每個類別展開成一欄

'Town'即使不列在categorical裡，程式仍會保留原始Town欄位，因為建滑動窗口時要依Town分組。      
==========================================================================
'''
FEATURE_SETS = {
    #全部特徵
    'full': {
        'continuous': ['Case_Count', 'RT', 'Rainfall', 'AvgTemp', 'TempRange',
                       'AvgHumidity', 'SunshineHours', 'RainfallHours', 'BI',
                       'CI', 'HI', 'LI', 'AI', 'PI', 'Con100HH'],
        'binary': ['Medicine'],
        'categorical': ['Town', 'Month'],
    },
    #精簡版：經permutation importance + 9折rolling CV確認後保留的特徵
    'reduced': {
        'continuous': ['Case_Count', 'RT'],
        'binary': [],
        'categorical': [],
    },
}

FEATURE_SET = 'reduced'

if FEATURE_SET not in FEATURE_SETS:
    raise ValueError(f"FEATURE_SET 必須是 {list(FEATURE_SETS)} 其中之一，目前是 {FEATURE_SET!r}")

CONTINUOUS_FEATURES = FEATURE_SETS[FEATURE_SET]['continuous']
BINARY_FEATURES = FEATURE_SETS[FEATURE_SET]['binary']
CATEGORICAL_FEATURES = FEATURE_SETS[FEATURE_SET]['categorical']

#模型目錄
SAVE_DIR = f'saved_models_{FEATURE_SET}'

#模型檔名
MODEL_FILENAME = (f'dengue_lstm_h{HIDDENSIZE}'
                   f'_w{WINDOWSIZE}'
                   f'_g{GAMMA}'
                   f'_{FEATURE_SET}.pth')
