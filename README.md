# 台南登革熱疫情時空預測模型
[視覺化網站](https://dengue-tainan.streamlit.app/)
本專案結合流行病學理論與深度學習，針對具備高度時間依賴性與資料極度不平衡特性的登革熱疫情，建構具備早期預警價值的時空預測模型。

## 核心技術與特色 
* **不平衡數據處理 (Imbalanced Data Learning)**：導入 **Focal Loss** 取代傳統 **Cross Entropy**，有效解決極端高風險樣本僅佔 1.88% 的痛點。
* **嚴謹特徵工程 (Feature Engineering)**：運用 **Permutation Importance** 進行壓力測試，將 18 項環境變數精簡為 4 項核心特徵 (Case_Count, RT, Month, Town)。
* **防洩漏驗證機制 (Data Leakage Prevention)**：實作 **Walk-forward 滾動式交叉驗證 (Rolling CV)**，確保時間序列模型評估的絕對客觀性。

## 模型成效 
* **模型架構**：雙層 Stacked LSTM (Hidden Size: 64, Window Size: 4)
* **獨立測試集盲測 (2023-2025)**：
  * 整體 Macro F1-score: **0.88**
  * 高風險擴散期 (Level 2) Recall: **0.75**

## 專案結構 
* `/LSTM`: 包含 PyTorch LSTM 架構與 Focal Loss 損失函數實作。
* `/data`: 清理後之特徵工程輸出範例資料。
* `/app` : 視覺化相關資料。

