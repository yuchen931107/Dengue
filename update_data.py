import datetime
from get_cases_data import get_cases
from get_latest_weatrer import get_weather
from get_invest_data import get_invest

if __name__ == "__main__":
    print("開始更新資料")
    print("="*50)
    
    #1.病例數更新
    print("(1/3)抓取最新確診病例\n")
    cases_df = get_cases()
    
    #2.weather更新
    print("(2/3)抓取近期氣象觀測\n")
    #用台南車站測試
    tainan_lat = 22.9997    
    tainan_lon = 120.2270
    tainan_lat_test = 23.038386  
    tainan_lon_test = 120.2367
    start_str = "2010-01-01"
    today = datetime.date.today()
    #將end_day設為前兩天
    end_str = (today - datetime.timedelta(days=2)).strftime("%Y-%m-%d")
    weather_df = get_weather(tainan_lat, tainan_lon, start_str, end_str)
    weather_df_test=get_weather(tainan_lat_test, tainan_lon_test, start_str, end_str)
    #3.調查資料更新
    print("(3/3)抓取病媒蚊調查紀錄\n")
    mos_df = get_invest()
    
    print("="*50)
    print("所有資料皆已更新完畢！")


