import datetime
from get_cases_data import get_cases
from get_latest_weather_data import get_weather
from get_invest_data import get_invest

if __name__ == "__main__":
    print("開始更新資料")
    print("="*50)
    
    #1.病例數更新
    print("(1/3)抓取最新確診病例\n")
    cases_df = get_cases()
    
    #2.weather更新
    print("(2/3)抓取近期氣象觀測\n")
    get_weather()
    
    #3.調查資料更新
    print("(3/3)抓取病媒蚊調查紀錄\n")
    mos_df = get_invest()
    
    print("="*50)
    print("所有資料皆已更新完畢！")


