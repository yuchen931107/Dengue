import pandas as pd

def Dengue_dataset():
    raw_url = "https://raw.githubusercontent.com/yuchen931107/Dengue/refs/heads/data-storage/Tainan_Dengue_ML.csv"
    df = pd.read_csv(raw_url)
    print("資料集:\n")
    df.info()
    return df
