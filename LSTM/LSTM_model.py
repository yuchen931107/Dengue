import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from LSTM_loader import dengue_dataloader
from set_seed import set_seed
set_seed(1234)
'''
非函數可調參數:
學習率:lr=0.001
神經網路層數:num_layers=2
防作弊斷線率:dropout=0.3
smoothed_weights[i]*k
'''
# ==========================================
# 1.核心架構定義區 (模型大腦 & 計分板)
# ==========================================
'''
簡易套件寫法:
model = nn.Sequential(
    nn.LSTM(input_size=16, hidden_size=128),
    nn.Dropout(0.3),
    nn.Linear(128, 3)
)
'''

'''
nn.LSTM： AI的記憶區。會按照時間順序（連續 windowsize 週）讀取資料，把前幾週的氣候、蚊蟲資訊轉化成內部的記憶。
nn.Dropout(0.3)：這是一個聰明的防作弊機制。它會在訓練時隨機把AI大腦裡30%的神經元關機。這會逼迫 AI不要過度依賴某些特定的特徵。
nn.Linear：最後一層的分類器，把 LSTM 整理好的複雜記憶，濃縮成3個數字（代表預測3個等級的機率）。
'''
class DengueLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(DengueLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        final_state = out[:, -1, :]
        final_state = self.dropout(final_state)
        predictions = self.fc(final_state)
        return predictions
'''
alpha (偏心權重)：這是前面在 DataLoader裡算出來、用來對付資料不平衡的 weight。
gamma (專注度因子)：它的作用是：如果 AI 覺得這題很簡單，就自動把這題的配分降到極低；逼迫 AI 把所有的算力集中在一直學不會的困難題目上。
reduction：決定最後要把這個 batch的分數加總還是平均起來。
'''
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss


# ==========================================
# 2.訓練與存檔函式
# ==========================================
def train_model(train_loader, val_loader, dim, weight, device, 
                hiddensize=128, gamma=2, EPOCHS=50, PATIENCE=7, 
                windowsize=6, save_dir='saved_models'):
    
    print("\n=== 2. 開始模型訓練 (Training Loop) ===")
    
    #設定模型（登革熱疫情等級固定為 4 級：Level 0~3）
    model = DengueLSTM(input_size=dim, hidden_size=hiddensize, num_layers=2, num_classes=4).to(device)

    #宣告權重(from loader)
    weights = weight.to(device)

    #Focal Loss gamma參數可調整控制
    criterion = FocalLoss(alpha=weights, gamma=gamma)

    #當模型猜錯時，Adam 負責指導神經網路要怎麼修改參數lr:learning rate
    #0.001 是 Adam 優化器業界公認的最佳初始值
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf') 
    early_stop_counter = 0
    best_model_weights = None

    for epoch in range(EPOCHS):
        #【訓練】
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

        #【驗證】用 val_loader，test_loader 
        model.eval()  #設成驗證模式
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad(): #設定此迴圈不能更新權重，僅驗證
            for batch_X, batch_y in val_loader:                        
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
                
        #計算loss
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)              
        #印 macro F1，在不平衡資料下比 accuracy 有診斷意義
        macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        print(f'Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | Val Macro F1: {macro_f1:.4f}') 


        '''if 連續 PATIENCE 個 Epoch val_loss都沒變更好就觸發Early Stopping並回溯'''
        if avg_val_loss < best_val_loss:                                
            best_val_loss = avg_val_loss                               
            best_model_weights = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"\n觸發 Early Stopping！模型在 Epoch {epoch+1} 提早停止訓練。")
                break

    #訓練結束，載入最好的權重並存檔
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print("已將模型權重恢復至最佳狀態。")
        
        # 自動建立資料夾並儲存模型
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 檔名可以包含參數，方便未來做實驗對照
        save_path = os.path.join(save_dir, f'dengue_lstm_h{hiddensize}_w{windowsize}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"模型參數已成功儲存至：{save_path}")
        
        return save_path

    print("警告：訓練過程中沒有出現任何有效的 val_loss，模型未儲存。")
    return None


# ==========================================
# 3.評估函式
# ==========================================
def evaluate_model(model_path, test_loader, dim, hiddensize, device, testyear):
    print(f"\n=== 讀取模型記憶：{model_path} ===")
    
    #宣告空model（登革熱疫情等級固定為 4 級：Level 0~3）
    model = DengueLSTM(input_size=dim, hidden_size=hiddensize, num_layers=2, num_classes=4).to(device)
    
    #載入model
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() #開啟測試模式
    
    all_preds = []
    all_targets = []

    print(f"=== 開始進行 {testyear} 測試集預測 ===")
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    target_names = ['Level 0', 'Level 1', 'Level 2', 'Level 3']

    print("\n測試集分類報告 (Classification Report):")
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))

    #混淆矩陣
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title(f'Dengue Fever Confusion Matrix\n(Test Set: {testyear})', fontsize=14, pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()


# ==========================================
# 4.主流程：訓練 → 評估 一次跑完
# ==========================================
if __name__ == '__main__':
    batch = 64
    windowsize = 6
    testyear = 2023
    hiddensize = 128
    gamma = 2

    print("=== 1. 載入資料與環境設定 ===")
    train_loader, val_loader, test_loader, weight, dim = dengue_dataloader(
        window_size=windowsize, batch_size=batch, split_year=testyear
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #訓練並取得存檔路徑
    save_path = train_model(
        train_loader=train_loader, val_loader=val_loader, dim=dim,
        weight=weight, device=device, windowsize=windowsize, hiddensize=hiddensize, gamma=gamma
    )

    #訓練成功才接著評估，避免 save_path 是 None 導致報錯
    if save_path is not None:
        evaluate_model(
            model_path=save_path, test_loader=test_loader, dim=dim,
            hiddensize=hiddensize, device=device, testyear=testyear
        )
    else:
        print("因訓練未產生模型檔，跳過評估階段。")