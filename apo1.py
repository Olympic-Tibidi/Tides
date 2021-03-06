import pandas as pd
import datetime
import numpy as np

import streamlit as st
from st_aggrid import AgGrid

import pickle
import re

import time


import matplotlib.pyplot as plt
import os
import glob
import base64
from collections import defaultdict

from requests import get

import os

from collections import defaultdict

import time
import json
from bs4 import BeautifulSoup


import xml.etree.ElementTree as ET
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.dates import DayLocator, HourLocator,DateFormatter,date2num,num2date
import matplotlib.ticker as plticker
from matplotlib.animation import FuncAnimation
import metpy.calc as mpcalc
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'
import statistics
from itertools import count
st.set_option('deprecation.showPyplotGlobalUse', False)




st.set_page_config(layout='wide')






def get_data(date):
    
    begin_date=date
    
    def vectorize(direction,speed):
        Wind_Direction=direction
        Wind_Speed=speed
        wgu = 0.1*Wind_Speed * np.cos((270-Wind_Direction)*np.pi/180)
        wgv= 0.1*Wind_Speed*np.sin((270-Wind_Direction)*np.pi/180)
        return(wgu,wgv)

    def parse_angle(angle_str):
        angle= mpcalc.parse_angle(angle_str)
        angle=re.findall(f'\d*\.?\d?',angle.__str__())[0]
        return float(angle)
    def get_weather():
        weather=defaultdict(int)
        headers = { 
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36', 
                'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 
                'Accept-Language' : 'en-US,en;q=0.5', 
                'Accept-Encoding' : 'gzip', 
                'DNT' : '1', # Do Not Track Request Header 
                'Connection' : 'close' }
        url ='https://api.weather.gov/gridpoints/SEW/117,51/forecast/hourly'
        #url='https://api.weather.gov/points/47.0379,-122.9007'   #### check for station info with lat/long
        durl='https://api.weather.gov/alerts?zone=WAC033'
        response = get(url,headers=headers)
        desponse=get(durl)
        data = json.loads(response.text)
        datan=json.loads(desponse.text)
        #print(data)

        for period in data['properties']['periods']:
            date=datetime.datetime.strptime(period['startTime'],'%Y-%m-%dT%H:%M:%S-07:00')
            #date_f=dt.datetime.strftime(dt.datetime.strptime(period['startTime'],'%Y-%m-%dT%H:%M:%S-08:00'),"%b-%d %H:%M")
            #date=f'{date_f} {period["name"]}' 
            #print(date)
            #print(period)
            weather[date]={'Wind_Direction':f'{period["windDirection"]}','Wind_Speed':f'{period["windSpeed"]}'}

        wind_forecast=pd.DataFrame.from_dict(weather,orient='index')
        wind_forecast.Wind_Speed=[int(re.findall(f'\d+',i)[0]) for i in wind_forecast.Wind_Speed.values]
        wind_forecast['Vector']=[vectorize(parse_angle(i),j) for i,j in zip(wind_forecast.Wind_Direction.values,wind_forecast.Wind_Speed.values)]

        return wind_forecast

    def find_tyde(stat,begin_date):
        station='9446484' if stat=='TAC' else '9447130'


        
        begin_date=begin_date
        begin_date=datetime.datetime.strftime(begin_date,'%Y%m%d %H:%M')

       
        now=datetime.datetime.now()
      

        resp = get(f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={begin_date}&range=24&station={station}&datum=MLLW&product=predictions&units=english&time_zone=lst&application=ports_screen&format=xml')


        #resp = get(f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={begin_date}&end_date={end_date}&station={station}&datum=MLLW&product=predictions&units=english&time_zone=lst&application=ports_screen&format=xml')
        with open('tides.xml', 'wb') as f:
            f.write(resp.content)


        tree = ET.parse('tides.xml')
        root = tree.getroot()

        lst=[(datetime.datetime.strptime(pr.attrib['t'],'%Y-%m-%d %H:%M'),float(pr.attrib['v'])) for pr in root.findall('pr')][:-1]


        return lst


    lst=find_tyde("SEA",begin_date)
    df = pd.DataFrame(lst, columns =['Time', 'Height'])
    wind_forecast=get_weather()


    wind_forecast['Florred']=[pd.to_datetime(i).floor('H') for i in wind_forecast.index.values]
    df['Florred']=[pd.to_datetime(i).floor('H') for i in df.Time.values]
    merged=pd.merge(df,wind_forecast,on='Florred')
    # wind_forecast=wind_forecast[:25]
    status=["Rising" if merged.Height.values[i+1]>merged.Height.values[i] else
                  'Falling' for i in range(len(merged.Height.values)-1)]
    addition="Rising" if merged.Height.values[-1]>merged.Height.values[-2] else "Falling"
    status.append(addition)
    merged['Status']=status

    level=[0]*(len(merged.Status))

    for i in range(len(merged.Status)-1):
        if merged.Status[i]=="Falling":
            if merged.Status[i+1]=='Rising':
                level[i+1]="Low Tide"
        if merged.Status[i]=="Rising":
            if merged.Status[i+1]=='Falling':
                level[i+1]="High Tide"
    merged['Level']=level

    filtered_peaks=merged[merged['Level']=='High Tide']
    filtered_lows=merged[merged['Level']=='Low Tide']
    high_tides=[(i,j) for i,j in zip(filtered_peaks.Time.values,filtered_peaks.Height.values)]
    low_tides=[(i,j) for i,j in zip(filtered_lows.Time.values,filtered_lows.Height.values)]
    high_tides_pd=[pd.to_datetime(i[0]) for i in high_tides]
    low_tides_pd=[pd.to_datetime(i[0]) for i in low_tides]

    high_tides_times=[date2num(i[0]) for i in high_tides]
    low_tides_times=[date2num(i[0]) for i in low_tides]

    k=[(pd.to_datetime(m),n,o,p) for m,n,o,p in zip(merged.Time.values,merged.Wind_Direction,merged.Wind_Speed,
                            merged.Vector.values)]
       
    
    x=[pd.to_datetime(i) for i in merged.Time.values]
    y=[i for i in merged.Height.values]

    #k=[(pd.to_datetime(m),n,o,p) for m,n,o,p in zip(merged.Time.values,merged.Wind_Direction.values,temp.Wind_Speed.values,
          #                      temp.Vector.values)]
    tuple_data=k,x,y,high_tides,low_tides,high_tides_times,low_tides_times,high_tides_pd,low_tides_pd
    return tuple_data

    
    





def get_schedule():
    df=pd.read_excel('schedule.xlsx')
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df.set_index('Vessel',drop=True,inplace=True)
    return df




def get_weather_frame():


    weather=defaultdict(int)
    url ='https://api.weather.gov/gridpoints/SEW/117,51/forecast'
    #url='https://api.weather.gov/points/47.0379,-122.9007'   #### check for station info with lat/long
    durl='https://api.weather.gov/alerts?zone=WAC033'
    headers = { 
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36', 
        'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 
        'Accept-Language' : 'en-US,en;q=0.5', 
        'Accept-Encoding' : 'gzip', 
        'DNT' : '1', # Do Not Track Request Header 
        'Connection' : 'close' }

    response = get(url,headers=headers).json()

    #print(response)
    for period in response['properties']['periods']:
        try:
            date=datetime.datetime.strptime(period['startTime'],'%Y-%m-%dT%H:%M:%S-08:00')
        except:
            date=datetime.datetime.strptime(period['startTime'],'%Y-%m-%dT%H:%M:%S-07:00')
        date_f=datetime.datetime.strftime(datetime.datetime.strptime(period['startTime'],'%Y-%m-%dT%H:%M:%S-07:00'),"%b-%d")
        date=f'{date_f} {period["name"]}' 
        #print(date)
        #print(period)
        weather[date]={'Temp':period['temperature'],'Wind':f'{period["windDirection"]}-{period["windSpeed"]}',
                        'Forecast':period['shortForecast'],'Detail':period['detailedForecast']}


    weather_forecast=pd.DataFrame.from_dict(weather,orient='index')
    weather_forecast.drop('Detail',axis=1,inplace=True)
    return weather_forecast


def ciz_la(tuple_data):
    plt.rcdefaults()

    k,x,y,high_tides,low_tides,high_tides_times,low_tides_times,high_tides_pd,low_tides_pd=tuple_data[0],tuple_data[1],tuple_data[2],tuple_data[3],tuple_data[4],tuple_data[5],tuple_data[6],tuple_data[7],tuple_data[8]
        ###############col_map = plt.get_cmap('tab20')

    fig, ax1 = plt.subplots(2,1,figsize=(16,12),gridspec_kw={'height_ratios': [8,15]},sharex=True)
    fig.patch.set_facecolor('blue')
    fig.patch.set_alpha(0.05)



    d=ax1[1].plot(x, y, linewidth=3,alpha=0.9,zorder=1)
    ax1[1].set_ylim(-3,15)
    now=datetime.datetime.now()-datetime.timedelta(hours=8) 
    nowel=date2num(datetime.datetime.now()-datetime.timedelta(hours=8) )        ###############            STREAMLIT PROBLEM
    xt = ax1[1].get_xticks()
    #print(xt)
    if now>=begin_date:          ###############            STREAMLIT PROBLEM
        xt=np.append(xt,nowel)
    #print(xt)
    xtl=xt.tolist()
    #xtl[-1]="Here is 1.5"
    ax1[1].set_xticks(xt)
    ax1[1].set_xticklabels(num2date(xtl))


    ax1[1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    # set formatter
    ax1[1].xaxis.set_major_formatter(mdates.DateFormatter('%b,%d-%H:%M'))
    # set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()

    now_lbl=f" Time Now: {datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=7) ,'%H:%M')}"              ###############            STREAMLIT PROBLEM


    if now>=begin_date:                                  ###############            STREAMLIT PROBLEM
        ax1[1].vlines(now, ax1[1].get_ylim()[0],ax1[1].get_ylim()[1], 'r')


    if now>=begin_date:                  ###############            STREAMLIT PROBLEM
        ax1[1].text(
        now,14,
        now_lbl,
        horizontalalignment='center',
        color='black',
        bbox ={'facecolor':'green',
                'alpha':0.2, 'pad':5},
        weight='bold',
        style='italic',
        size=13
    )

    for i in high_tides_times:
        xt=np.append(xt,(i))
    for i in low_tides_times:
        xt=np.append(xt,i)
    xtl=xt.tolist()
    ax1[1].set_xticks(xt)
    ax1[1].set_xticklabels(num2date(xtl))

    nl = '\n'
    high_tide1_label=f"High Tide : {round(high_tides[0][1],1)}ft @{nl}{datetime.datetime.strftime(high_tides_pd[0],'%H:%M')}"
    ax1[1].vlines(high_tides_times[0], ax1[1].get_ylim()[0],high_tides[0][1], 'g',ls="--")
    ax1[1].text(
        high_tides_times[0],high_tides[0][1]-3,
        high_tide1_label,
        horizontalalignment='center',
        color='black',
        bbox ={'facecolor':'yellow',
                'alpha':0.2, 'pad':5},
        weight='bold',
        style='italic',
        size=11
    )
    if len(high_tides_times)>1:
        high_tide2_label=f"High Tide : {round(high_tides[1][1],1)}ft @{nl}{datetime.datetime.strftime(high_tides_pd[1],'%H:%M')}"
        ax1[1].vlines(high_tides_times[1], ax1[1].get_ylim()[0],high_tides[1][1], 'g',ls="--")
        ax1[1].text(
        high_tides_times[1],high_tides[1][1]-3,
        high_tide2_label,
        horizontalalignment='center',
        color='black',
        bbox ={'facecolor':'yellow',
                'alpha':0.2, 'pad':5},
        weight='bold',
        style='italic',
        size=11
            )
    low_tide1_label=f"Low Tide : {round(low_tides[0][1],1)}ft @{nl}{datetime.datetime.strftime(low_tides_pd[0],'%H:%M')}"
    ax1[1].vlines(low_tides_times[0], ax1[1].get_ylim()[0],low_tides[0][1], 'g',ls="--")
    ax1[1].text(
        low_tides_times[0],low_tides[0][1]+3,
        low_tide1_label,
        horizontalalignment='center',
        color='black',
        bbox ={'facecolor':'yellow',
                'alpha':0.2, 'pad':5},
        weight='bold',
        style='italic',
        size=11
    )
    if len(low_tides_times)>1:
        low_tide2_label=f"Low Tide : {round(low_tides[1][1],1)}ft @{nl}{datetime.datetime.strftime(low_tides_pd[1],'%H:%M')}"
        ax1[1].vlines(low_tides_times[1], ax1[1].get_ylim()[0],low_tides[1][1], 'g',ls="--")
        ax1[1].text(
        low_tides_times[1],low_tides[1][1]+3,
        low_tide2_label,
        horizontalalignment='center',
        color='black',
        bbox ={'facecolor':'yellow',
                'alpha':0.2, 'pad':5},
        weight='bold',
        style='italic',
        size=11
    )






    ax1[0].grid(False)
    ax1[1].grid(True,alpha=0.4,color='black')


    #### PUT numbers at regular intervals ####

    for i in range(len(x))[::20]:
        ax1[1].text(
        x[i] ,
        y[i] + 0.6,
        round(y[i],1),  
        size=12,
        horizontalalignment='center',
        color='black',
        weight='bold'
    )

    ax1[1].tick_params(axis="x", labelsize=12) 

    


    ax1[1].set_xlabel('24 HOUR PERIOD',size=12, labelpad=15, color='#333333', weight='bold')
    ax1[1].set_ylabel('TIDE HEIGHT IN FEET',size=12, labelpad=15, color='#333333', weight='bold')
    begin_date_str=datetime.datetime.strftime(begin_date,"%b,%d %H:%M")
    to_date_str=datetime.datetime.strftime(begin_date+datetime.timedelta(hours=24),"%b,%d %H:%M")
    fig.suptitle(f'SEATTLE TIDES and WIND FORECAST'+'\n'+'\n'+f'{begin_date_str} to {to_date_str}'
                , size=14, color='#333333',
                weight='bold')



    img2=plt.imread('t5.jpg')
    img3=plt.imread('sunny.jpg')

    ax1[1].imshow(img2,zorder=0, extent=[ax1[1].get_xlim()[0], ax1[1].get_xlim()[1],
                                        ax1[1].get_ylim()[0], ax1[1].get_ylim()[1]],aspect='auto',alpha=0.3)
    ax1[0].imshow(img3,zorder=0, extent=[ax1[0].get_xlim()[0], ax1[0].get_xlim()[1],
                                        ax1[0].get_ylim()[0], 30],aspect='auto',alpha=0.3)



    ###########    WIND PLOT    ############

    wind_speeds=[i[2] for i in k]
    ax1[0].set_ylim(0,30)
    for i in k[::20]:

        ax1[0].quiver(i[0],ax1[0].get_ylim()[1]/2,i[3][0],i[3][1],scale=10,pivot='middle') #headwidth=1,headlength=2
        ax1[0].text(
            i[0],
            ax1[0].get_ylim()[1]/100,
            f'{i[1]}\n{i[2]} miles',
        size=12,   
        horizontalalignment='center',
        color='black',
        weight='bold')
    if now>=begin_date:                  ###############            STREAMLIT PROBLEM
        ax1[0].text(
        now,ax1[0].get_ylim()[1]/1.1,
        now_lbl,
        horizontalalignment='center',
        color='black',
        bbox ={'facecolor':'green',
                'alpha':0.2, 'pad':5},
        weight='bold',
        style='italic',
        size=13
    )

    if now>=begin_date:                      ###############            STREAMLIT PROBLEM    
        ax1[0].vlines(now, ax1[0].get_ylim()[0],ax1[0].get_ylim()[1], 'r')


    ax1[0].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    # set formatter
    ax1[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()

    ax1[0].set_ylabel('WIND SPEED/DIRECTION',size=12, labelpad=15, color='#333333', weight='bold')


    ax1[0].tick_params(axis='both', which='both', labelsize=11, labelbottom=True,rotation=45)
    ax1[1].tick_params(axis='both', which='both', labelsize=11)


    ax1[0].patch.set_facecolor('orange')
    ax1[0].patch.set_alpha(0.1)
    ax1[1].patch.set_facecolor('yellow')
    ax1[1].patch.set_alpha(0.1)


    #fig.tight_layout()
    plt.savefig('plot.png')
    plt.show()
        
def check_container_no(container):
    liste=[i for i in container]
    a=0
    b=0
    if len(container)<11 or '?' in container:
        return('Container Number Missing Digit ')
    for i,j in enumerate([k for k in container][:4]):
        a+=2**i*letter_dict[j]
    for i,j in enumerate([k for k in container][4:-1]):
        z=2**(i+4)
        b+=(int(z)*int(j))
    check=(a+b)-int((a+b)/11)*11 
    if check==int(container[-1]):
        return('Container Number Legitimate')
    else:
        return('Container Number Is Wrong')

def guess_missing_number(container):
    
    liste=[i for i in container]
    a=0
    b=0
    
    def calculate_letters(a):
        for i,j in enumerate([k for k in container][:4]):
            a+=(2**i)*letter_dict[j]
        return a
   
    
    def figure_letter(a,b):
    
        for i,j in enumerate([k for k in container][:4]):
            if j=='?':
                #print(i)
                unknown_index=i
                #print(unknown_index)
                unknown_calculation=2**(unknown_index)
                #print(unknown_calculation)
            else:
                z=2**(i)
                a+=(int(z)*int(letter_dict[j]))
        #print(f'a is {a}')
        x=0
        pos=[]
        for q in letter_dict.keys():
            #print(q)
            m=(2**unknown_index)*letter_dict[q]-(int(((2**unknown_index)*letter_dict[q]+a+b)/11)*11)-(int(container[-1])-a-b)
            if m ==0:
                pos.append(q)
        if unknown_index==3 and "U" in pos:
            pos=["U"]
        return pos
    
    def calculate_number(b):
        for i,j in enumerate([k for k in container][4:-1]):
            z=2**(i+4)
            b+=(int(z)*int(j))
        return b
    
    def figure_number(a,b):
    
        for i,j in enumerate([k for k in container][4:-1]):
            if j=='?':
                #print(i)
                unknown_index=i+4
                #print(unknown_index)
                unknown_calculation=2**(unknown_index)
                #print(unknown_calculation)
            else:
                z=2**(i+4)
                b+=(int(z)*int(j))
        #print(f'b is {b}')
        x=0
        for q in range(10):
            m=(2**unknown_index)*q-(int(((2**unknown_index)*q+a+b)/11)*11)-(int(container[-1])-a-b)
            if m ==0:
                return q
            
    if '?' not in container:
        return check_container_no(container)
    if '?' in container[4:-1]:
        a=0
        b=0
        a=calculate_letters(a)
        target=figure_number(a,b)
    if '?' in container[:4]:
        target_pool=[]
        a=0
        b=0
        b=calculate_number(b)
        pos=figure_letter(a,b)
        for i in pos:
            possibility=container.replace('?',str(i))
            if check_container_no(possibility)=='Container Number Legitimate':
                target_pool.append(i)
            else:
                continue
        target=target_pool
        if target==['U']:
            target='U'
            
        print(f'target pool is {target}')
        
        trial=container
        print(trial)
        for i in target:
            z=trial.replace('?',str(i))
            #print(z)
            if z[:3] not in owner_codes:
                target.remove(i)
                
    if container[-1]=='?':
        b=0      
        n=calculate_number(b)
        a=0
        l=calculate_letters(a)
        target=n+l-(int((n+l)/11)*11)
    if type(target) is list:
        return f"Missing Letter can be one of {target},and possible container numbers: {[container.replace('?',str(i)) for i in target]}"
        
    return f"Missing Digit is {target} and container number is {container.replace('?',str(target))}"
    
owner_codes= pickle.load(open("owner_codes.dat", "rb"))
letter_dict= pickle.load(open("bic_letters.dat", "rb"))

begin_date=datetime.datetime.now()-datetime.timedelta(hours=7)                     ###############            STREAMLIT PROBLEM

data=()
if data not in st.session_state:
    tuple_data=get_data(begin_date)
    ciz_la(tuple_data)
    st.session_state.data=tuple_data



selection = st.sidebar.radio("Choose Report",[
    'Wind/Tides','Weather Forecast','Ship Schedule','Container Check'])



###### WEATHER FORECAST   ######

if selection == "Weather Forecast":
    st.write(get_weather_frame())
    with st.expander("See explanation/Source"):
        st.write("""
         Data compiled from National Weather Service API instantly
     """)


######### SHIP SCHEDULE   ######

if selection== "Ship Schedule":
    st.write(get_schedule())
    with st.expander("See explanation/Source"):
        st.write("""
         Ship Schedule compiled from nwseaportalliance.com periodically.
         Tide Information is calculated from NOAA Tides Web API based on ETA date/time""")



if selection=='Wind/Tides':
    
    # d = st.sidebar.date_input(
    #         "Choose Begin Date", begin_date.date())
    # t = st.sidebar.time_input('Choose Begin Time',begin_date.time())
    button = st.sidebar.button("Refresh Data")

    if button:
        
        ###begin_date=datetime.datetime.combine(d,t)  #    LETS NOT CHOOOSE FOR NOW
        begin_date=datetime.datetime.now()-datetime.timedelta(hours=7) 
        tuple_data=get_data(begin_date)
        ciz_la(tuple_data)
        st.session_state.data=tuple_data
    with st.container():
        
        k,x,y=st.session_state.data[0],st.session_state.data[1],st.session_state.data[2]
        from PIL import Image
        image = Image.open('plot.png')

        st.image(image, caption='Chart Compiled By Afsin Yilmaz')

if selection == "Container Check":
    st.subheader("Verifies a container number, or finds the missing digit/letter based on an algorithm reverse engineered from the BIC convention algorithm. For missing digits it produces the one and only possibility. For missing letters it produces up to 3 possibilities.") 
    st.subheader("Enter Container Number to verify (use capital letters for Letters; Use question mark ('?') in place of missing digit or letter. Finally click 'CHECK' button")
    st.markdown('E.g : Container Number : SZLU9313?14 (with a missing/failed to read digit) or S?LU9313014 (with a missing/failed to read Letter)')
    container = st.text_input('Container Number')
    button = st.button("CHECK")
    if button :
        st.title(guess_missing_number(container))
