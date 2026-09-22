import torch
from get_Dengue import Dengue_dataset
from LSTM_rolling import make_fold_data, train_fold
from LSTM_importance import build_feature_groups, permutation_importance, plot_importance
from config import WINDOWSIZE

'''
**********************************************************************
用途：
  挑特徵時，正式版的val（單純前一年）幾乎沒有Level2樣本（通常只有1筆），
  拿來算permutation importance跟拿來做Early Stopping一樣沒有意義。

  解法：借用LSTM_rolling.py裡「val=2015」那一折（train=2011-2014，
  val=2015，191筆Level2），單獨訓練一個模型，用這一折的val來算
  特徵重要性——概念上跟挑GAMMA/HIDDENSIZE/WINDOWSIZE時「重點看第2折」
  是同一件事，只是這裡把「比較不同超參數」換成「比較有無某個特徵」。

  這裡訓練用的HIDDENSIZE/GAMMA，直接沿用config.py裡最終選定的值，
  只有WINDOWSIZE用config.py的值來建這一折的窗口，其餘（train_fold用的
  超參數）也是讀config.py，確保「用來挑特徵的模型」跟「最終要用的模型」
  架構一致，重要性排序才有參考價值。

  這支程式跑完之後，看哪些特徵可以拿掉，回頭去改
  LSTM_preprocessing.py的continuous_features，
  再用正式設定（VAL_RANGES=None）重新訓練最終模型。
**********************************************************************
'''

VAL_YEAR_FOR_IMPORTANCE = 2015   #唯一有足夠Level2樣本可以拿來看重要性的年份

if __name__ == '__main__':
    seed = 1234
    n_repeats = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== 用 val={VAL_YEAR_FOR_IMPORTANCE} 這一折訓練模型（僅供特徵重要性分析用） ===")
    df = Dengue_dataset()
    Xtr, ytr, Xva, yva, all_features = make_fold_data(df, VAL_YEAR_FOR_IMPORTANCE, WINDOWSIZE)
    dim = Xtr.shape[2]

    model, best_epoch, macro_f1, per_class = train_fold(Xtr, ytr, Xva, yva, dim, seed)
    print(f"停在第{best_epoch}輪 | val macro F1={macro_f1:.4f} | 各級F1={per_class.round(3).tolist()}\n")

    groups = build_feature_groups(all_features)
    print(f"共 {len(groups)} 組特徵（含 Town、Month 分組），開始計算重要性...\n")

    baseline_f1, importances = permutation_importance(
        model, Xva, yva, groups, device, n_repeats=n_repeats
    )
    plot_importance(importances, f"Val={VAL_YEAR_FOR_IMPORTANCE}（挑特徵專用，非最終結果）")

    print("\n=== 提示 ===")
    print("F1 掉幅接近 0 或為負值的特徵，代表打亂它幾乎不影響模型表現，")
    print("屬於可以考慮移除的候選變數；掉幅越大代表模型越依賴該特徵，應保留。")
    print("\n※ 這個模型只是為了看特徵重要性而訓練，不是最終模型，也不會存檔。")
    print("   決定好要保留哪些特徵後，記得回到LSTM_preprocessing.py調整，")
    print("   再用config.py的正式設定（VAL_RANGES=None）重新訓練最終模型。")