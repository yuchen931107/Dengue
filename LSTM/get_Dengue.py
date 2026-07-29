import pandas as pd

def Dengue_dataset():
    url = "https://raw.githubusercontent.com/yuchen931107/Dengue/refs/heads/data-storage/Tainan_Dengue_ML.csv"
    df = pd.read_csv(url)
    #df = pd.read_csv("Tainan_Dengue_ML.csv")
    return df
df=Dengue_dataset()
df.info()
