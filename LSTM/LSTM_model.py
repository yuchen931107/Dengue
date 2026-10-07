import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from LSTM_loader import dengue_dataloader
from set_seed import set_seed
from config import (WINDOWSIZE, HIDDENSIZE, BATCH, GAMMA, LR, SPLIT_YEAR, TESTYEAR_LABEL,
                    SAVE_DIR, MODEL_FILENAME, NUM_CLASSES, LEVEL_NAMES, FIXED_EPOCHS,
                    NUM_LAYERS,DROPOUT,PATIENCE,MAX_EPOCH)

from config import SEED
set_seed(SEED)
'''
nn.LSTM： AI的記憶區。會按照時間順序（連續 windowsize 週）讀取資料，把前幾週的氣候、蚊蟲資訊轉化成內部的記憶。
nn.Dropout(0.3)：這是一個防作弊機制。它會在訓練時隨機把AI大腦裡30%的神經元關機。這會逼迫 AI不要過度依賴某些特定的特徵。
nn.Linear：把 LSTM 整理好的記憶，用 softmax 濃縮成3個機率
'''
# ==========================================
# 1.核心架構定義區 (模型大腦 & Focal loss)
# ==========================================
class DengueLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(DengueLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        final_state = out[:, -1, :]  # 取所有 batch、最後一個時間步、全部 hidden_size
        final_state = self.dropout(final_state)
        predictions = self.fc(final_state)
        return predictions

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        #先用 log_softmax 算出不含 alpha 的乾淨 pt，做完降權，最後才乘上類別權重。
        logp = F.log_softmax(inputs, dim=1) #將 logit 轉換為機率 pi=e^zi/sum(e^zj)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()                
        focal_loss = ((1 - pt) ** self.gamma) * (-logpt)        
        if self.alpha is not None:
            focal_loss = focal_loss * self.alpha[targets]       
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

# ==========================================
# 2.訓練與存檔函式
# ==========================================
def train_model(train_loader, dim, weight, device, epochs, val_loader=None, 
                hiddensize=HIDDENSIZE,patience=7, gamma=GAMMA, save_dir=SAVE_DIR):
    """
    若有提供 val_loader，則使用 Early Stopping (最高輪數為 epochs)；
    若未提供，則執行固定輪數 (Fixed Epochs) 訓練。
    """
    mode_str = "Early Stopping" if val_loader is not None else "固定輪數"
    print(f"\n=== 2. 開始模型訓練 ({mode_str} 模式，最高輪數 = {epochs}) ===")

    #設定模型
    model = DengueLSTM(input_size=dim, hidden_size=hiddensize, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES).to(device)

    #宣告權重(alpha)
    weights = weight.to(device)

    #Focal Loss
    criterion = FocalLoss(alpha=weights, gamma=gamma)

    #0.001 是 Adam 優化器業界公認的最佳初始值
    optimizer = optim.Adam(model.parameters(), lr=LR)

    #記錄每個 epoch 的 train loss 及 val loss。
    train_loss_history = []
    val_loss_history = []
    
    # Early stopping 初始化
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    stall = 0

    for epoch in range(epochs):
        #【訓練階段】
        model.train()  #設成訓練模式
        train_loss = 0.0    
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()               #把上一個batch的錯誤紀錄擦掉，避免記憶混亂。
            outputs = model(batch_X)            #模型嘗試猜測出 level
            loss = criterion(outputs, batch_y)  #算出Loss 分數
            loss.backward()                     #往回推導
            #把更新的步伐限制在 1.0 以內，確保學習過程平穩
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        #【驗證與 Early Stopping 階段】
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)
            print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
            
            # 檢查是否破紀錄
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                stall = 0
                # 複製並儲存當下最佳的模型權重
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stall += 1
                if stall >= patience:
                    print(f"\n連續 {patience} 輪 Val Loss 未下降，啟動 Early Stopping！")
                    print(f"訓練中斷於第 {epoch+1} 輪，還原至最佳的第 {best_epoch} 輪權重。")
                    break
        else:
            # Fixed Epoch 的模式
            print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f}')
    
    # 若有啟用 Early Stopping，訓練結束後必須將模型參數讀檔還原至 best_epoch
    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        
          
    #【繪製 Loss 曲線圖】
    plt.figure(figsize=(8, 5))
    epochs_ran = range(1, len(train_loss_history) + 1)
    plt.plot(epochs_ran, train_loss_history, label='Train Loss', color='#4C72B0', linewidth=2)

    if val_loader is not None:
        val_epochs_ran = range(1, len(val_loss_history) + 1)
        plt.plot(val_epochs_ran, val_loss_history, label='Val Loss', color='#C44E52', linewidth=2)
        # 畫出垂直虛線標示最佳輪數
        plt.axvline(best_epoch, color='gray', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.title('Train vs Validation Loss Curve (Early Stopping)')
    else:
        plt.title('Train Loss Curve (Fixed Epochs)')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 存檔並回傳路徑
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, MODEL_FILENAME)
    torch.save(model.state_dict(), save_path)
    print(f"模型參數已成功儲存至：{save_path}")
    return save_path


# ==========================================
# 3.評估函式
# ==========================================
def evaluate_model(model_path, test_loader, dim, hiddensize, device, testyear):
    print(f"\n=== 讀取模型記憶：{model_path} ===")
    
    #宣告空model（等級數量統一由 config.py 的 NUM_CLASSES 決定）
    model = DengueLSTM(input_size=dim, hidden_size=hiddensize, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES).to(device)
    
    #載入model
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() #開啟測試模式
    
    all_preds = []
    all_targets = []

    print(f"=== 開始進行推論：{testyear} ===")
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    target_names = LEVEL_NAMES
    label_ids = list(range(NUM_CLASSES))

    print(f"\n分類報告 (Classification Report) — {testyear}:")
    report_dict = classification_report(all_targets, all_preds, labels=label_ids,
                                         target_names=target_names,
                                         zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    report_df = report_df.drop(index=['accuracy', 'weighted avg'], errors='ignore')
    print(report_df.round(2))

    #製作混淆矩陣
    cm = confusion_matrix(all_targets, all_preds, labels=label_ids)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title(f'Dengue Fever Confusion Matrix\n({testyear})', fontsize=14, pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()


# ==========================================
# 4.主流程：訓練 → 評估 
# ==========================================
if __name__ == '__main__':               
    print("=== 1. 載入資料與環境設定 ===")
    
    train_loader, val_loader, test_loader, weight, dim = dengue_dataloader(
        window_size=WINDOWSIZE, batch_size=BATCH, split_year=SPLIT_YEAR
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #決定要傳入的 Epoch 數量與印出的提示文字
    actual_epochs = MAX_EPOCH if val_loader is not None else FIXED_EPOCHS
    mode_info = f"Early Stopping (最高 {MAX_EPOCH} 輪)" if val_loader is not None else f"固定輪數 ({FIXED_EPOCHS} 輪)"
    print(f"--- 當前訓練設定: {mode_info} ---")

    #訓練並取得存檔路徑
    save_path = train_model(
        train_loader=train_loader, dim=dim, weight=weight, device=device, 
        epochs=actual_epochs, val_loader=val_loader, patience=PATIENCE, 
        hiddensize=HIDDENSIZE, gamma=GAMMA, save_dir=SAVE_DIR
    )

    #訓練成功接著評估
    if save_path is not None:
        evaluate_model(
            model_path=save_path, test_loader=test_loader, dim=dim,
            hiddensize=HIDDENSIZE, device=device, testyear=f"Test Set: {TESTYEAR_LABEL}"
        )
    else:
        print("因訓練未產生模型檔，跳過評估階段。")
