import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from LSTM_preprocessing import preprocessing
from LSTM_loader import create_time_windows
from LSTM_model import DengueLSTM

'''
**********************************************************************
用途：檢查每個特徵對模型表現的影響力（Permutation Importance）
原理：把某個特徵的值在樣本間打亂（其他特徵、時間結構不變），
      重新預測一次，看 Macro F1 掉多少。
      掉得越多 → 代表模型越依賴這個特徵 → 越重要。

注意：Town_*、Month_* 是 One-Hot 展開後的欄位，
      單獨打亂其中一欄沒有意義，所以會把同一類別的所有欄位
      綁在一起、同時打亂，評估「Town 整組」「Month 整組」的重要性。
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
def permutation_importance(model, X_test, y_test, groups, device, n_repeats=5, seed=1234):
    rng = np.random.default_rng(seed)

    baseline_preds = predict_all(model, X_test, device)
    baseline_f1 = f1_score(y_test, baseline_preds, average='macro', zero_division=0)
    print(f"=== Baseline Macro F1（未打亂）: {baseline_f1:.4f} ===\n")

    importances = {}
    for name, idxs in groups.items():
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm_order = rng.permutation(len(X_perm))
            #把這組特徵（所有時間步）整條從別的樣本換過來，其他特徵維持不變
            X_perm[:, :, idxs] = X_perm[perm_order][:, :, idxs]

            perm_preds = predict_all(model, X_perm, device)
            perm_f1 = f1_score(y_test, perm_preds, average='macro', zero_division=0)
            drops.append(baseline_f1 - perm_f1)

        importances[name] = (float(np.mean(drops)), float(np.std(drops)))
        print(f"{name:20s} | F1 掉幅: {np.mean(drops):+.4f} ± {np.std(drops):.4f}")

    return baseline_f1, importances

#畫成橫向長條圖，由重要到不重要排序
def plot_importance(importances, testyear):
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
    plt.title(f'Feature Importance（Permutation, Test: {testyear}）', fontsize=13, pad=12)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    windowsize = 6
    testyear = 2023
    hiddensize = 128
    n_repeats = 5  # 想要結果更穩定可調大，但跑的時間會變長

    model_path = f'saved_models/dengue_lstm_h{hiddensize}_w{windowsize}.pth'

    print("=== 載入資料 ===")
    train_df, val_df, test_df, all_features = preprocessing(split_year=testyear)
    X_test, y_test = create_time_windows(test_df, all_features, 'RT_level', windowsize)
    dim = X_test.shape[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== 讀取模型記憶：{model_path} ===")
    model = DengueLSTM(input_size=dim, hidden_size=hiddensize, num_layers=2, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    groups = build_feature_groups(all_features)
    print(f"共 {len(groups)} 組特徵（含 Town、Month 分組），開始計算重要性...\n")

    baseline_f1, importances = permutation_importance(
        model, X_test, y_test, groups, device, n_repeats=n_repeats
    )
    plot_importance(importances, testyear)

    #如果掉幅接近 0 甚至是負的，代表這個特徵對模型幾乎沒貢獻，是可以考慮拿掉的候選
    print("\n=== 提示 ===")
    print("F1 掉幅接近 0 或為負值的特徵，代表打亂它幾乎不影響模型表現，")
    print("屬於可以考慮移除的候選變數；掉幅越大代表模型越依賴該特徵，應保留。")