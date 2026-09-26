import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from LSTM_preprocessing import preprocessing_full
from LSTM_loader import build_all_windows
from LSTM_model import DengueLSTM
from LSTM_rolling import make_fold_data, train_fold
from config import WINDOWSIZE, HIDDENSIZE, SPLIT_YEAR, SAVE_DIR, MODEL_FILENAME, VAL_RANGES, NUM_CLASSES, PURGE

'''
**********************************************************************
用途：檢查每個特徵對模型表現的影響力（Permutation Importance）
原理：把某個特徵的值在樣本間打亂（其他特徵、時間結構不變），
      重新預測一次，看 Macro F1 掉多少。
      掉得越多 → 代表模型越依賴這個特徵 → 越重要。

注意：Town_*、Month_* 是 One-Hot 展開後的欄位，
      單獨打亂其中一欄沒有意義，所以會把同一類別的所有欄位
      綁在一起、同時打亂，評估「Town 整組」「Month 整組」的重要性。

兩種模式（MODE，見下方 __main__）：
  'fold2015'：借用 LSTM_rolling.py 裡「val=2015」那一折（train=2011-2014，
              val=2015，有充足的Level2樣本），單獨訓練一個臨時模型（不存檔）。
              用途：挑特徵階段的決策依據。正式版的val（單純前一年）幾乎沒有
              Level2樣本，不能拿來判斷特徵要不要保留，只有這一折的val有
              足夠訊號可以參考。

  'test'    ：讀取 SAVE_DIR/MODEL_FILENAME 已經訓練好、定案的正式模型，
              對 test 集算重要性。用途：模型都定案、訓練完之後，拿來寫進
              報告解讀最終模型依賴哪些特徵——這時特徵已經選完了，不是用
              這次結果去做任何進一步的特徵取捨。
**********************************************************************
'''

#把 One-Hot 展開的欄位歸併成同一組，其餘連續特徵各自獨立一組
def build_feature_groups(all_features):
    groups = {}
    for i, col in enumerate(all_features):
        if col.startswith('Town_'):
            groups.setdefault('Town（區域類別）', []).append(i)
        elif col.startswith('Month_'):
            groups.setdefault('Month（月份類別）', []).append(i)
        else:
            groups[col] = [i]
    return groups

#批次跑模型預測，避免一次把整個 test set 塞進 GPU/CPU
def predict_all(model, X, device, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted.cpu().numpy())
    return np.concatenate(preds)

#核心：對每組特徵重複打亂 n_repeats 次，取平均 F1 掉幅當作重要性分數
def permutation_importance(model, X_eval, y_eval, groups, device, n_repeats=5, seed=1234):
    rng = np.random.default_rng(seed)

    baseline_preds = predict_all(model, X_eval, device)
    baseline_f1 = f1_score(y_eval, baseline_preds, average='macro', zero_division=0,
                            labels=list(range(NUM_CLASSES)))
    print(f"=== Baseline Macro F1（未打亂）: {baseline_f1:.4f} ===\n")

    importances = {}
    for name, idxs in groups.items():
        drops = []
        for _ in range(n_repeats):
            X_perm = X_eval.copy()
            perm_order = rng.permutation(len(X_perm))
            #把這組特徵（所有時間步）整條從別的樣本換過來，其他特徵維持不變
            X_perm[:, :, idxs] = X_perm[perm_order][:, :, idxs]

            perm_preds = predict_all(model, X_perm, device)
            perm_f1 = f1_score(y_eval, perm_preds, average='macro', zero_division=0,
                                labels=list(range(NUM_CLASSES)))
            drops.append(baseline_f1 - perm_f1)

        importances[name] = (float(np.mean(drops)), float(np.std(drops)))
        print(f"{name:20s} | F1 掉幅: {np.mean(drops):+.4f} ± {np.std(drops):.4f}")

    return baseline_f1, importances

#畫成橫向長條圖，由重要到不重要排序
def plot_importance(importances, label):
    sorted_items = sorted(importances.items(), key=lambda kv: kv[1][0], reverse=True)
    names = [k for k, _ in sorted_items]
    means = [v[0] for _, v in sorted_items]
    stds = [v[1] for _, v in sorted_items]

    colors = ['#C44E52' if m < 0 else '#4C72B0' for m in means]

    plt.figure(figsize=(9, max(4, len(names) * 0.35)))
    plt.barh(names, means, xerr=stds, color=colors)
    plt.gca().invert_yaxis()
    plt.axvline(0, color='gray', linewidth=0.8)
    plt.xlabel('Macro F1 掉幅（baseline - 打亂後）')
    plt.title(f'Feature Importance（Permutation, {label}）', fontsize=13, pad=12)
    plt.tight_layout()
    plt.show()


#=====================================================================
#兩種模式各自負責「準備model + 準備X_eval/y_eval + 準備all_features」
#=====================================================================

def _prepare_fold2015(seed=1234):
    """訓練一個臨時模型（僅供特徵重要性分析用，不存檔），並回傳val=2015的評估資料"""
    from get_Dengue import Dengue_dataset

    VAL_YEAR_FOR_IMPORTANCE = 2015   #唯一有足夠Level2樣本可以拿來看重要性的年份
    print(f"=== 用 val={VAL_YEAR_FOR_IMPORTANCE} 這一折訓練模型（僅供特徵重要性分析用） ===")
    df = Dengue_dataset()
    Xtr, ytr, Xva, yva, all_features = make_fold_data(df, VAL_YEAR_FOR_IMPORTANCE, WINDOWSIZE)

    model, best_epoch, macro_f1, per_class = train_fold(Xtr, ytr, Xva, yva, Xtr.shape[2], seed)
    print(f"停在第{best_epoch}輪 | val macro F1={macro_f1:.4f} | 各級F1={per_class.round(3).tolist()}\n")

    label = f"Val={VAL_YEAR_FOR_IMPORTANCE}（挑特徵專用，非最終結果）"
    return model, Xva, yva, all_features, label


def _prepare_test(device):
    """讀取已經訓練好、定案的正式模型，並回傳test集的評估資料"""
    model_path = f'{SAVE_DIR}/{MODEL_FILENAME}'
    testyear = SPLIT_YEAR   #這裡要用數字（餵給 preprocessing_full 判斷年份），不是顯示用的文字標籤

    print("=== 載入資料 ===")
    #跟主程式(dengue_dataloader)使用完全相同的方式建構test集，
    #確保這裡算出的樣本數/Baseline F1，跟LSTM_model.py的test分類報告可以直接對照
    #【修正】val_ranges必須跟訓練模型時用的一致，否則train資料範圍不同
    #→StandardScaler算出的平均數/標準差不同→同一批test資料被餵進去的數字就不一樣
    full_df, all_features, split_labels = preprocessing_full(split_year=testyear, val_ranges=VAL_RANGES)
    X_all, y_all, target_idx = build_all_windows(full_df, all_features, 'RT_level', WINDOWSIZE,
                                                  split_labels=split_labels, purge=PURGE)
    test_mask = split_labels.loc[target_idx].values == 'test'
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    print(f"=== 讀取模型記憶：{model_path} ===")
    model = DengueLSTM(input_size=X_test.shape[2], hidden_size=HIDDENSIZE, num_layers=2,
                        num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    label = str(testyear)
    return model, X_test, y_test, all_features, label


if __name__ == '__main__':
    #==============================================================
    #MODE='fold2015'：挑特徵階段用，看val=2015的重要性分數（開發用，非最終結果）
    #MODE='test'    ：特徵都定案、模型訓練完之後，看test的重要性分數（寫進報告用）
    #==============================================================
    MODE = 'fold2015'
    n_repeats = 5  # 想要結果更穩定可調大，但跑的時間會變長
    seed = 1234    # fold2015模式下，換這個數字重跑可以檢查排序穩不穩

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODE == 'fold2015':
        model, X_eval, y_eval, all_features, label = _prepare_fold2015(seed)
    elif MODE == 'test':
        model, X_eval, y_eval, all_features, label = _prepare_test(device)
    else:
        raise ValueError("MODE 只能是 'fold2015' 或 'test'")

    groups = build_feature_groups(all_features)
    print(f"共 {len(groups)} 組特徵（含 Town、Month 分組），開始計算重要性...\n")

    baseline_f1, importances = permutation_importance(
        model, X_eval, y_eval, groups, device, n_repeats=n_repeats, seed=seed
    )
    plot_importance(importances, label)

    #如果掉幅接近 0 甚至是負的，代表這個特徵對模型幾乎沒貢獻，是可以考慮拿掉的候選
    print("\n=== 提示 ===")
    print("F1 掉幅接近 0 或為負值的特徵，代表打亂它幾乎不影響模型表現，")
    print("屬於可以考慮移除的候選變數；掉幅越大代表模型越依賴該特徵，應保留。")
    if MODE == 'fold2015':
        print("\n※ 這個模型只是為了看特徵重要性而訓練，不是最終模型，也不會存檔。")
        print("   決定好要保留哪些特徵後，記得回到LSTM_preprocessing.py與LSTM_rolling.py")
        print("   同步調整特徵清單，再用config.py的正式設定重新訓練最終模型。")