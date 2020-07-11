import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def Clean(path):
    df = pd.read_excel(path)
    df = df[['DIR', 'YR', 'MO', 'DAY', 'HR', 'MIN', 'SEC', 'ANAME', 'ON', 'OFF', 'DLMILES', "LOAD", "LAT", "LONG"]]
    #cleaning time and date
    df.loc[df.HR >= 24,"DAY"] += 1
    df.loc[df.DAY == 32, "MO"] += 1
    df.loc[df.DAY == 32, "DAY"] =1
    df.loc[(df.MO == 4) & (df.DAY==31), ("DAY", "MO")] = (1, 5)
    df.loc[(df.MO == 6) & (df.DAY==31), ("DAY", "MO")] = (1, 7)
    df.loc[df.HR == 24, "HR"] = 0
    df.loc[df.HR == 25, "HR"] = 1
    df.loc[df.HR == 26, "HR"] = 2
    df.loc[df.HR == 27, "HR"] = 3
    df.loc[df.HR == 28, "HR"] = 4
    df.loc[df.HR == 29, "HR"] = 5
    #calculating the duration
    df['date'] = df['MO'].astype(str) + '/' + df['DAY'].astype(str) + '/' + df['YR'].astype(str) + ' ' + (df['HR'].astype(str)) + ':' + (df['MIN'].astype(str)) + ':' + (df['SEC'].astype(str))
    df['date']= pd.to_datetime(df.date)
    df['new_date'] = df['date'].shift(1)
    df['duration']= ((df['date'] - df['new_date']).dt.total_seconds())
    #new indexing of on and off board
    df['new_on']= df['ON'].shift(1)
    df['new_off']=df['OFF'].shift(1)
    df.loc[df.DIR =="OUTBOUND", "DIR"] = 1
    df.loc[df.DIR =="INBOUND", "DIR"] = 0
    df = df.fillna(df.mean())
    df = df[['duration', 'new_on','new_off', "LOAD","DLMILES", "DIR"]]
    # df = pd.read_csv("cleaned/"+i).drop("Unnamed: 0", 1)
    df = df.loc[(df.duration <= 100) , :]
    df = df.loc[df.duration >= 10, :]
    return df


