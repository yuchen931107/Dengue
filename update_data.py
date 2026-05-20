import datetime
from get_cases_data import get_cases
from get_latest_weather_data import get_weather
from get_invest_data import get_invest
from get_population_data import get_population
from Count_RT import count_RT

if __name__ == "__main__":
    print("開始更新資料")
    print("="*50)
    
    print("抓取最新確診病例\n")
    cases_df = get_cases()
    
    print("抓取近期氣象觀測\n")
    get_weather()
    
    print("抓取病媒蚊調查紀錄\n")
    mos_df = get_invest()

    print("抓取病媒蚊調查紀錄\n")
    get_population()

    print("計算RT")
    count_RT()
    
    print("所有資料皆已更新完畢！")


