import pandas as pd
import datetime
import numpy as np

import streamlit as st
from st_aggrid import AgGrid


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
            date=datetime.datetime.strptime(period['startTime'],'%Y-%m-%dT%H:%M:%S-08:00')
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
    k=[(pd.to_datetime(m),n,o,p) for m,n,o,p in zip(merged.Time.values,merged.Wind_Direction,merged.Wind_Speed,
                            merged.Vector.values)]
       
    
    x=[pd.to_datetime(i) for i in merged.Time.values]
    y=[i for i in merged.Height.values]

    #k=[(pd.to_datetime(m),n,o,p) for m,n,o,p in zip(merged.Time.values,merged.Wind_Direction.values,temp.Wind_Speed.values,
          #                      temp.Vector.values)]
    return k,x,y

    
    





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
        date=datetime.datetime.strptime(period['startTime'],'%Y-%m-%dT%H:%M:%S-08:00')
        date_f=datetime.datetime.strftime(datetime.datetime.strptime(period['startTime'],'%Y-%m-%dT%H:%M:%S-08:00'),"%b-%d")
        date=f'{date_f} {period["name"]}' 
        #print(date)
        #print(period)
        weather[date]={'Temp':period['temperature'],'Wind':f'{period["windDirection"]}-{period["windSpeed"]}',
                        'Forecast':period['shortForecast'],'Detail':period['detailedForecast']}


    weather_forecast=pd.DataFrame.from_dict(weather,orient='index')
    weather_forecast.drop('Detail',axis=1,inplace=True)
    return weather_forecast


def ciz_la(k,x,y):
    plt.rcdefaults()


        ###############col_map = plt.get_cmap('tab20')

    fig, ax1 = plt.subplots(2,1,figsize=(16,12),gridspec_kw={'height_ratios': [8,15]},sharex=True)
    fig.patch.set_facecolor('blue')
    fig.patch.set_alpha(0.05)



    d=ax1[1].plot(x, y, linewidth=3,alpha=0.9,zorder=1)

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

    now_lbl=f" Time Now: {datetime.datetime.strftime(datetime.datetime.now()-datetime.timedelta(hours=8) ,'%H:%M')}"              ###############            STREAMLIT PROBLEM


    if now>=begin_date:                                  ###############            STREAMLIT PROBLEM
        ax1[1].vlines(now, ax1[1].get_ylim()[0],ax1[1].get_ylim()[1], 'r')


    if now>=begin_date:                  ###############            STREAMLIT PROBLEM
        ax1[1].text(
        now,ax1[1].get_ylim()[1]/1.2,
        now_lbl,
        horizontalalignment='center',
        color='black',
        bbox ={'facecolor':'green',
                'alpha':0.2, 'pad':5},
        weight='bold',
        style='italic',
        size=13
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

    ax1[1].set_ylim(-1,13)


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
        



begin_date=datetime.datetime.now()-datetime.timedelta(hours=8)                     ###############            STREAMLIT PROBLEM

data=()
if data not in st.session_state:
    k,x,y=get_data(begin_date)
    ciz_la(k,x,y)
    st.session_state.data=(k,x,y)



selection = st.sidebar.radio("Choose Report",[
    'Wind/Tides','Weather Forecast'])



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

    button = st.button("Refresh Data")
    d = st.date_input(
            "Choose Begin Date", begin_date.date())
    t = st.time_input('Choose Begin Time',begin_date.time())
    
    if button:
        
        begin_date=datetime.datetime.combine(d,t)
        k,x,y=get_data(begin_date)
        ciz_la(k,x,y)
        st.session_state.data=(k,x,y)
    with st.container():
        
        k,x,y=st.session_state.data[0],st.session_state.data[1],st.session_state.data[2]
        from PIL import Image
        image = Image.open('plot.png')

        st.image(image, caption='Chart Compiled By Afsin Yilmaz')



