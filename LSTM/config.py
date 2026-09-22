'''
**********************************************************************
共用設定檔：所有腳本共用的參數，只要改這裡一個地方就好，
不用再跑去 LSTM_model.py、LSTM_importance.py 分別手動對齊數字。

只有需要一致才放進來（例如影響模型結構、影響存檔檔名的參數）；
不需要跨檔案保持一致的參數（例如訓練用的 EPOCHS/PATIENCE），
繼續留在各自腳本裡就好，不用硬塞進這裡。
**********************************************************************
'''

WINDOWSIZE = 6      #滑動時間窗大小（週數），影響模型輸入形狀 → 存檔/讀檔都要對齊
HIDDENSIZE = 64     #LSTM 隱藏層大小，影響模型結構 → 存檔/讀檔都要對齊
BATCH = 64           #訓練用的batch size
GAMMA = 2.5          #Focal Loss 的 gamma 參數
LR = 0.001           #Adam optimizer的學習率

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

SAVE_DIR = 'saved_models'
MODEL_FILENAME = f'dengue_lstm_h{HIDDENSIZE}_w{WINDOWSIZE}.pth'