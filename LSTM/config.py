'''
**********************************************************************
共用設定檔：所有腳本共用的參數，只要改這裡一個地方就好，
不用再跑去 LSTM_model.py、LSTM_importance.py 分別手動對齊數字。

只有需要一致才放進來（例如影響模型結構、影響存檔檔名的參數）；
不需要跨檔案保持一致的參數（例如訓練用的 EPOCHS/PATIENCE），
繼續留在各自腳本裡就好，不用硬塞進這裡。
**********************************************************************
'''
SEED = 1234         # 隨機種子碼，確保實驗可重現
WINDOWSIZE = 4      #滑動時間窗大小（週數），影響模型輸入形狀 → 存檔/讀檔都要對齊

HIDDENSIZE = 64     #LSTM 隱藏層大小，影響模型結構 → 存檔/讀檔都要對齊
BATCH = 64           #訓練用的batch size
GAMMA = 2         #Focal Loss 的 gamma 參數
LR = 0.001           #Adam optimizer的學習率

'''
正式訓練時使用的固定輪數 (由移除 RT 後的 Rolling CV 9 折 best_epoch 中位數決定)
設為整數時，最終模型會以 2011-2022 全部資料訓練，不另外切 2022 當 validation。
If 設定為 None 則使用 Early Stopping
'''
FIXED_EPOCHS = 12


#分級方式（三級）：
#  Level 0：RT <= 0.01      → 無有效訊號
#  Level 1：0.01 < RT <= 1  → 有真實訊號，但低於流行病學閾值
#  Level 2：RT > 1          → 高於閾值（Rt=1是Cori et al. 2013框架下唯一有實質意義的門檻），疫情成長中

NUM_CLASSES = 3
LEVEL_NAMES = ['Level 0', 'Level 1', 'Level 2']

SPLIT_YEAR = 2023           #test = Year >= SPLIT_YEAR
TESTYEAR_LABEL = "2023-2025"  #只用來顯示在圖表標題/文字上

#正式定案版本維持 None（單純用 split_year 前一年當val）；
#開發/調參數階段如果需要看Level2/3表現，可以暫時在各自腳本裡覆蓋成
#[('2015-10-01','2015-12-31'), ('2022-01-01','2022-12-31')] 之類的自訂區間，
#但記得：這只能拿來看訓練有沒有跑順，不能拿來跟正式版本比較數字（train樣本量不同）
VAL_RANGES = None

#embargo（切分邊界留空檔）：丟掉「輸入週跨越 train/val/test 邊界」的窗口。
#不開的話，val/test 邊界附近會出現跟 train 高度重疊（6週輸入裡有5週一樣）的窗口，
#模型等於考過類似的題目，驗證/測試分數會虛高。建議維持 True。
PURGE = True

#==========================================================================
#特徵設定：所有「用哪些特徵建模」的決定都集中在這裡，
#LSTM_preprocessing.py、LSTM_rolling.py、LSTM_importance.py 都改讀這裡，
#只要切 FEATURE_SET 或改下面的清單，所有腳本就會同步，不用各自手動對齊。
#
#三種特徵型態，處理方式不同：
#  continuous  ：連續型，會做標準化（StandardScaler，只用train fit）
#  binary      ：本身就是0/1，不標準化，直接輸入模型
#  categorical ：類別型，會做One-Hot編碼（只用train fit），每個類別展開成一欄
#
#注意：'Town'即使不列在categorical裡，程式仍會保留原始Town欄位，
#      因為建滑動窗口時要依Town分組，這跟它是不是模型特徵無關。
#==========================================================================
FEATURE_SETS = {
    #全部特徵（18組：15連續 + 1二元 + Town + Month）
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
        'categorical': ['Town', 'Month'],
    },
    #想再測別的組合（例如拿掉Town）：在這裡加一個條目，再把FEATURE_SET改成它的名字即可
}

FEATURE_SET = 'reduced'

if FEATURE_SET not in FEATURE_SETS:
    raise ValueError(f"FEATURE_SET 必須是 {list(FEATURE_SETS)} 其中之一，目前是 {FEATURE_SET!r}")

CONTINUOUS_FEATURES = FEATURE_SETS[FEATURE_SET]['continuous']
BINARY_FEATURES = FEATURE_SETS[FEATURE_SET]['binary']
CATEGORICAL_FEATURES = FEATURE_SETS[FEATURE_SET]['categorical']

#模型目錄與正式特徵集同步，避免不同特徵實驗的模型互相覆蓋。
SAVE_DIR = f'saved_models_{FEATURE_SET}'

#檔名帶上GAMMA跟FEATURE_SET，避免不同超參數/特徵組合的模型互相覆蓋，比對時也一眼分得出來
MODEL_FILENAME = (f'dengue_lstm_h{HIDDENSIZE}'
                   f'_w{WINDOWSIZE}'
                   f'_g{GAMMA}'
                   f'_{FEATURE_SET}.pth')
