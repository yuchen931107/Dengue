import os
import random
import numpy as np
import torch
#確保每次實驗結果一致
def set_seed(seed):
    # 1. 固定 Python 內建的隨機性
    random.seed(seed)
    
    # 2. 固定系統環境的 Hash 隨機性
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. 固定 NumPy 的隨機性 (例如計算 Class Weights 時)
    np.random.seed(seed)
    
    # 4. 固定 PyTorch CPU 的隨機性
    torch.manual_seed(seed)
    
    # 5. 固定 PyTorch GPU (CUDA) 的隨機性
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果未來你用多張顯卡
        
    # 6. 強制 GPU 底層運算一致 (注意：這可能會讓訓練速度稍微變慢一點點，但保證結果 100% 一致)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



