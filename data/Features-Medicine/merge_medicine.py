import datetime
import glob
import os
import re
import pandas as pd

# 臺南市 37 行政區代碼與名稱對照表
DISTRICT_MAP = {
    "67000010": "新營區",
    "67000020": "鹽水區",
    "67000030": "白河區",
    "67000040": "柳營區",
    "67000050": "後壁區",
    "67000060": "東山區",
    "67000070": "麻豆區",
    "67000080": "下營區",
    "67000090": "六甲區",
    "67000100": "官田區",
    "67000110": "大內區",
    "67000120": "佳里區",
    "67000130": "學甲區",
    "67000140": "西港區",
    "67000150": "七股區",
    "67000160": "將軍區",
    "67000170": "北門區",
    "67000180": "新化區",
    "67000190": "善化區",
    "67000200": "新市區",
    "67000210": "安定區",
    "67000220": "山上區",
    "67000230": "玉井區",
    "67000240": "楠西區",
    "67000250": "南化區",
    "67000260": "左鎮區",
    "67000270": "仁德區",
    "67000280": "歸仁區",
    "67000290": "關廟區",
    "67000300": "龍崎區",
    "67000310": "永康區",
    "67000320": "東區",
    "67000330": "南區",
    "67000340": "北區",
    "67000350": "安南區",
    "67000360": "安平區",
    "67000370": "中西區",
}


def read_csv_with_encoding(file_path):
    for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"無法解析編碼: {file_path}")


def parse_to_date(val, roc_year):
    """將日期轉換為西元 date 物件"""
    val_str = str(val).strip().split(".")[0]
    if not val_str or val_str.lower() in ["nan", "none", ""]:
        return None
    try:
        # 含分隔符號 (例如 107/1/25 或 2018-01-25)
        if "/" in val_str or "-" in val_str:
            parts = re.split(r"[/-]", val_str)
            if len(parts) == 3:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                if y < 1900:  # 民國年
                    y += 1911
                return datetime.date(y, m, d)
            elif len(parts) == 2:
                y = int(roc_year) + 1911
                return datetime.date(y, int(parts[0]), int(parts[1]))

        # 純數字 (例如 125, 1019, 1070125, 20180125)
        if val_str.isdigit():
            if len(val_str) in [3, 4]:
                m = int(val_str[:-2])
                d = int(val_str[-2:])
                y = int(roc_year) + 1911
                return datetime.date(y, m, d)
            elif len(val_str) == 7:  # 民國年月日
                return datetime.date(
                    int(val_str[:3]) + 1911,
                    int(val_str[3:5]),
                    int(val_str[5:]),
                )
            elif len(val_str) == 8:  # 西元年月日
                return datetime.date(
                    int(val_str[:4]), int(val_str[4:6]), int(val_str[6:])
                )
    except Exception:
        return None
    return None


def process_file(file_path):
    # 從檔名解析民國年
    match = re.search(r"medicine-(\d+)", file_path)
    roc_year = match.group(1) if match else "107"

    df = read_csv_with_encoding(file_path)
    df.columns = df.columns.astype(str).str.strip()

    # 1. 取得區域 (Town)
    town_series = None
    code_cols = [c for c in df.columns if "行政區域代碼" in c or "區代碼" in c]
    if code_cols:
        town_series = (
            df[code_cols[0]]
            .astype(str)
            .str.split(".")
            .str[0]
            .str.strip()
            .map(DISTRICT_MAP)
        )

    if town_series is None or town_series.isnull().all():
        name_cols = [
            c for c in df.columns if c in ["區", "行政區", "區別", "區域", "Town"]
        ]
        if name_cols:
            town_series = df[name_cols[0]]

    # 2. 取得日期並轉成所屬週的週一 (Week，格式 YYYY-MM-DD)
    date_cols = [c for c in df.columns if "日期" in c or "Date" in c]
    date_series = df[date_cols[0]].apply(lambda x: parse_to_date(x, roc_year))
    week_series = date_series.apply(
        lambda d: (d - datetime.timedelta(days=d.weekday())).strftime(
            "%Y-%m-%d"
        )
        if d
        else None
    )

    sub_df = pd.DataFrame({"Week": week_series, "Town": town_series})
    return sub_df.dropna(subset=["Week", "Town"])


def merge_medicine_features(
    folder_path=".", output_filename="medicine_features.csv"
):
    pattern = os.path.join(folder_path, "medicine-*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print("找不到符合 'medicine-*.csv' 的檔案，請確認檔案放置路徑。")
        return

    dfs = [process_file(f) for f in files]
    full_df = pd.concat(dfs, ignore_index=True)

    # 依照對齊後的 [Week, Town] 彙總計算「噴藥次數」
    result_df = full_df.groupby(["Week", "Town"]).size().reset_index(name="次數")

    # 輸出單一特徵整合檔
    output_path = os.path.join(folder_path, output_filename)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 50)
    print(f"合併完成！輸出檔案: {output_path}")
    print(f"總筆數: {len(result_df)} 筆")
    print(f"欄位: {list(result_df.columns)}")
    print("=" * 50)
    print("前 5 筆預覽：")
    print(result_df.head())

    return result_df


if __name__ == "__main__":
    merge_medicine_features()