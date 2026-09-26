import pandas as pd

def Dengue_dataset():
    url = "https://raw.githubusercontent.com/yuchen931107/Dengue/refs/heads/data-storage/Tainan_Dengue_ML.csv"
    df = pd.read_csv(url)
    return df

