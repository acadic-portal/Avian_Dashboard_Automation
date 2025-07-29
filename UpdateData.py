# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:45:32 2025

@author: movaz
"""

import numpy as np
import datetime
import time
import pandas as pd
from pytrends.request import TrendReq
from pytrends import dailydata
from geopy.geocoders import Nominatim
from tqdm import tqdm
#from pygooglenews import GoogleNews as pgn
from GoogleNews import GoogleNews as gn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import scipy.stats as stt
import math
import os
from scipy import optimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
#import pycountry
import json
import geopandas as gpd
import time
import requests

from bs4 import BeautifulSoup as bs

#%%
# Collect Cases
xls = pd.ExcelFile('https://cfia-ncr.maps.arcgis.com/sharing/rest/content/items/715ece7ec0224cf7adb602cf83ee99f6/data')
cases = pd.read_excel(xls, 'HPAI_Wildlife_Dashboard')

province = {'Newfoundland and Labrador':'NL', 'Nova Scotia':'NS', 'Prince Edward Island':'PE', 'New Brunswick':'NB', 'British Columbia':'BC', 'Ontario':'ON', 'Quebec':'QC', 'Saskatchewan':'SK', 'Alberta':'AB', 'Manitoba':'MB', 'Yukon':'YK', 'Northwest Territories':'NT', 'Nunavut':'NU', 'Atlantic Ocean':'AO'}
cases['Province'] = [province[x] for x in cases['Province']]

cases.to_csv('cases.csv', index=False) # It will save the file on disk. We will not have it in the next run.

last = str(max(pd.to_datetime(cases['Collection Date'])))[:10]
end = datetime.datetime.now() #- datetime.timedelta(days= 1)
start = datetime.datetime.strptime(last, '%Y-%m-%d') - datetime.timedelta(days= 600)
end = end.strftime('%Y-%m-%d')
start = start.strftime('%Y-%m-%d')

start1 = start
complete_dates = [start1]
while pd.to_datetime(start1) < pd.to_datetime(last): #(end):
    start1 = datetime.datetime.strptime(start1, '%Y-%m-%d') + datetime.timedelta(days= 1)
    start1 = start1.strftime('%Y-%m-%d')
    complete_dates.append(start1)
    
#%%

# We will try Google Trends for 30 times, if it did not work, we will just move on.
s = datetime.datetime.strptime(start, '%Y-%m-%d')
e = datetime.datetime.strptime(end, '%Y-%m-%d')
gt_success = False
q = 0
while gt_success == False:
  try:
    # The function has a wait_time equal to 5. If it was taking a long time, increase the wait_time
    gt = dailydata.get_daily_data('/m/0292d3', s.year, s.month-1, e.year, e.month, geo = 'CA')#, verbose= False) #, wait_time=5)
#    print('ok')
    gt_success = True
 #   break
  except:
#    print('Failed')
    if q < 30:
        q = q + 1
        continue
    else:
        break

if gt_success == False:
  gt = pd.read_csv('https://acadic-portal.github.io/Avian/dummies/gt.csv')    
gt.to_csv('gt.csv')

#%%

# Collect Google News
cols = ['title', 'desc', 'date', 'datetime', 'link', 'img', 'media', 'site', 'reporter']
googlen = pd.DataFrame (data= {}, columns= cols)

googlen0 = pd.read_csv('https://acadic-portal.github.io/Avian/dummies/gn1.csv')
start0 = str(max(pd.to_datetime(googlen0['Date'])))[:10]

previous = 0
s = start0
q = 0
gn_success = True
while pd.to_datetime(s) <= pd.to_datetime(end):
  try:
    s = datetime.datetime.strptime(s, "%Y-%m-%d")
    next = s + datetime.timedelta(days= 1)
    s = s.strftime("%m/%d/%Y")
    next = next.strftime("%m/%d/%Y")
    if pd.to_datetime(next) > pd.to_datetime(end):
      next = datetime.datetime.strptime(end, "%Y-%m-%d") + datetime.timedelta(days= 1)
      next = next.strftime("%m/%d/%Y")

    googlenews_en = gn(lang='en', region='CA', encode='utf-8', start= s, end= next) # Apparently it is mm/dd/YYYY
    googlenews_en.get_news('avian')
    r1 = googlenews_en.results()
    r1 = pd.DataFrame.from_records (r1)
    googlenews_en.get_news('h5')
    r2 = googlenews_en.results()
    r2 = pd.DataFrame.from_records (r2)
    googlenews_en.get_news('bird flu')
    r3 = googlenews_en.results()
    r3 = pd.DataFrame.from_records (r3)
    googlenews_en.get_news('bird influenza')
    r4 = googlenews_en.results()
    r4 = pd.DataFrame.from_records (r4)
    avian1 = pd.concat([r1, r2, r3, r4])
    avian1['Language'] = ['English']*len(avian1)
    avian1['Province'] = ['Whole Country']*len(avian1)

    googlenews_fr = gn(lang='fr', region='CA', encode='utf-8', start= s, end= next) # Apparently it is mm/dd/YYYY
    googlenews_fr.get_news('aviaire')
    r1 = googlenews_fr.results()
    r1 = pd.DataFrame.from_records (r1)
    googlenews_fr.get_news('h5')
    r2 = googlenews_en.results()
    r2 = pd.DataFrame.from_records (r2)
    avian2 = pd.concat([r1, r2])
    avian2['Language'] = ['French']*len(avian2)
    avian2['Province'] = ['Whole Country']*len(avian2)

    googlen = pd.concat([googlen, avian1, avian2])

    if len(googlen) - previous > 50:
        googlen.to_csv ('gn1.csv', index= False)
#        print(len(googlen), s, next)
        previous = len(googlen)

    s = datetime.datetime.strptime(next, "%m/%d/%Y") #- datetime.timedelta(days= 1)
    s = s.strftime("%Y-%m-%d")
    q = 0
  except:
    if q < 15:
        q = q + 1
        continue
    else:
        gn_success = False
        break
    
if gn_success == True:
    googlen.to_csv ('gn1.csv', index= False)
    
    dates = {'janv.':'Jan', 'févr.':'Feb', 'f√©vr.':'Feb', 'mars':'Mar', 'avr.':'Apr', 'mai':'May', 'juin':'Jun', 'juil.':'Jul', 'août':'Aug','ao√ªt.':'Aug','sept.':'Sep',
             'oct.':'Oct','nov.':'Nov','déc.':'Dec', 'd√©c.':'Dec', }
    
    googlen = googlen.reset_index(drop=True)
    
    dt = []
    today = datetime.datetime.strptime(end, '%Y-%m-%d') # datetime.datetime.today()
    for i in googlen.index:
      try:
        if pd.isnull(googlen['datetime'][i]) == False:
          dt.append(str(googlen['datetime'][i])[:10])
        elif 'jours' in googlen['date'][i]:
          dummy = str(googlen['date'][i]).split(' ')
          try:
            dummy = (today - datetime.timedelta(days= int(dummy[3][:-6]))).strftime('%Y-%m-%d')
          except:
            dummy = (today - datetime.timedelta(days= int(dummy[3][:-7]))).strftime('%Y-%m-%d')
          dt.append(dummy)
        elif 'heure' in str(googlen['date'][i]) or 'hour' in str(googlen['date'][i]) or 'minute' in str(googlen['date'][i]):
          dt.append(today.strftime('%Y-%m-%d'))
        elif 'Hier' in str(googlen['date'][i]) or 'Yesterday' in str(googlen['date'][i]):
          dt.append((today - datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
        elif 'days' in str(googlen['date'][i]):
          dummy = str(googlen['date'][i]).split(' ')
          dt.append(int(dummy[0]))
        else:
          dummy = googlen['date'][i]
          if len(dummy) <= 9:
            dummy = dummy + ' ' + str(today.year)
      
          dayfirst=False
          for k in dates.keys():
            if k in dummy:
              dummy = dummy.replace(k, dates[k])
              dayfirst=True
              break
      
          dummy = str(pd.to_datetime(dummy, dayfirst=dayfirst))[:10]
          if pd.to_datetime(dummy) > pd.to_datetime(today.strftime('%Y-%m-%d')):
            dummy = str(today.year - 1) + dummy[4:]
          dt.append(dummy)
      except:
        dt.append(dt[-1]) # For some reason, it wasn't able find the date for this record, I will just put the value of the previous record for it.
        continue
    
    googlen['Date'] = dt
    #googlen0 = googlen0[googlen0['Date'] != start0]
    googlen = pd.concat([googlen0, googlen])
    googlen = googlen.drop_duplicates()
    googlen = googlen.drop(columns=['title','desc','date','datetime','link','img','site'])
    googlen.to_csv ('gn1.csv', index= False)
else:
    googlen = pd.read_csv('https://acadic-portal.github.io/Avian/dummies/gn1.csv')
    googlen.to_csv('gn1.csv', index=False)
    
#%%
# Reddit
reddit = pd.read_csv('https://acadic-portal.github.io/Avian/dummies/reddit2.csv')

cols = ['subreddit_id', 'approved_at_utc', 'author_is_blocked', 'comment_type', 'edited', 'mod_reason_by', 'banned_by', 'ups', 'num_reports',
        'author_flair_type', 'total_awards_received', 'subreddit','author_flair_template_id', 'likes', 'replies', 'user_reports', 'saved',
        'id', 'banned_at_utc', 'mod_reason_title', 'gilded', 'archived','collapsed_reason_code', 'no_follow', 'author', 'can_mod_post',
        'send_replies', 'parent_id', 'score', 'author_fullname','report_reasons', 'removal_reason', 'approved_by', 'all_awardings',
        'body', 'awarders', 'top_awarded_type', 'downs','author_flair_css_class', 'author_patreon_flair', 'collapsed',
        'author_flair_richtext', 'is_submitter', 'body_html', 'gildings','collapsed_reason', 'associated_award', 'stickied', 'author_premium',
        'can_gild', 'link_id', 'unrepliable_reason', 'author_flair_text_color','score_hidden', 'permalink', 'subreddit_type', 'locked', 'name',
        'created', 'author_flair_text', 'treatment_tags', 'created_utc','subreddit_name_prefixed', 'controversiality',
        'author_flair_background_color', 'collapsed_because_crowd_control','mod_reports', 'mod_note', 'distinguished']
#reddit = pd.DataFrame (data={}, columns=cols)
subr = ['Canada_sub', 'canadanews', 'canada', 'ontario','KingstonOntario', 'toronto', 'ottawa', 'Quebec', 'montreal', 'alberta',
        'Calgary', 'Edmonton', 'britishcolumbia', 'vancouver', 'vancouvercanada', 'British_Columbia', 'regina', 'saskatchewan', 'Manitoba',
        'winnipegnews', 'Winnipeg', 'nunavut', 'NovaScotia', 'NovaScotiaGardening', 'halifax', 'newbrunswickcanada', 'NewBrunswick',
        'newfoundland', 'Newfoundland_Labrador', 'princeedwardisland','PrinceEdwardCounty', 'Charlottetown','nunavut', 'Yukon',
        'NorthWestTerritories', 'CapeBreton', 'AtlanticCanada','canadanews','OttawaNews', 'OntarioNews', 'TorontoNews', 'QuebecNews',
        'AlbertaNews', 'VancouverNews','HalifaxNews']
#kword = ['bird', 'avian', 'poultry', 'farm', 'waterfowl', 'h5', 'aviaire', 'oiseau', 'sauvagine', 'sauvagin']
kword = ['"avian flu"','"avian influenza"','"bird flu"','"bird influenza"','avianinfluenza','birdflu','birdinfluenza','h5','hpai']

day = 86400
previous = 0
i = 0
en = (datetime.datetime.strptime(end, '%Y-%m-%d') + datetime.timedelta(days=1)).timestamp()
st = (datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=60)).timestamp()
r_success = True
q = 0
while i < len(subr):
    s = subr[i]
    j = 0
    while j < len(kword):
        k = kword[j]
        b = en #2024-07-16
        a = b - 180*day
        while b > st: #1420084800:
            try:
                myurl = 'https://api.pullpush.io/reddit/comment/search?q=' + k + '&subreddit=' + s + '&after=' + str(int(a)) + '&before=' + str(int(b))
                r = requests.get (myurl).json()
                if 'data' in list(r.keys()):
                    dummy = pd.DataFrame.from_records (r['data'])
                    if len(dummy) > 0:
                        date = [datetime.datetime.fromtimestamp(int(item)) for item in list(dummy['created_utc'])]
                        dummy['date'] = date
                        reddit = pd.concat([reddit, dummy])
 #                       print ('subreddit:',s, ', length:', len(dummy))
                    if len(reddit) - previous > 100:
                        # Please change this path to the path you want on your computer:
                        reddit.to_csv ('reddit2.csv', index= False)
                        previous = len(reddit)
#                        print('reddit1', len(reddit), ', last time-stamp:', list(dummy['created_utc'])[-1])
                    b = a #+ day
                    a = b - 180*day
                else:
 #                   print('sleep')
                    time.sleep(60)
                    continue
                q = 0
            except:
                if q < 5:
  #                  print(q, 'sleep')
                    q += 1
                    time.sleep(60)
                    continue
                else:
                    r_success = False
                    break

        j += 1
        if r_success == False:
            break
    i += 1
    if r_success == False:
        break
    
if r_success == True:
   reddit.to_csv ('reddit2.csv', index= False)
else:
  reddit = pd.read_csv('https://acadic-portal.github.io/Avian/dummies/reddit2.csv')
  reddit.to_csv ('reddit2.csv', index= False)

#%%
# GDELT
header = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
'Accept-Encoding': 'gzip, deflate, br, zstd',
'Accept-Language': 'en-US,en;q=0.9',
'Connection': 'keep-alive',
'Cookie': '_ga=GA1.1.1793398363.1734542457; _ga_ZZ0YC7MJ34=GS1.1.1734542457.1.0.1734542457.60.0.0',
'Host': 'api.gdeltproject.org',
'Referer': 'https://api.gdeltproject.org/api/v2/summary/summary?d=web&t=summary&k=%28avian+OR+%22bird+flu%22+OR+%22bird+influenza%22+OR+h5n1+OR+h5nx+OR+hpai+OR+%22highly+pathogenic+avian+influenza%22%29&ts=custom&sdt=20200101000000&edt=20241217235959&fsc=CA&svt=zoom&stc=yes&sta=list&c=1',
'Sec-Fetch-Dest': 'iframe',
'Sec-Fetch-Mode': 'navigate',
'Sec-Fetch-Site': 'same-origin',
'Upgrade-Insecure-Requests': '1',
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 OPR/114.0.0.0',
'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Opera";v="114"',
'sec-ch-ua-mobile': '?0',
'sec-ch-ua-platform': '"Windows"'}

s = start.replace('-','') + '000000'
e = end.replace('-','') + '000000'

myurl = 'https://api.gdeltproject.org/api/v2/doc/doc?format=html&startdatetime=' + s + '&enddatetime=' + e + '&query=(avian%20OR%20%22bird%20flu%22%20OR%20%22bird%20influenza%22%20OR%20%22highly%20pathogenic%20avian%20influenza%22%20OR%20hpai%20OR%20h5n1%20OR%20h5nx)%20sourcecountry:CA&mode=timelinevol&timezoom=yes'
gdelt_success = False
q = 0
while True:
    try:
        r2 = requests.get (myurl, headers = header)
        gdelt_success = True
        break
    except:
        q += 1
        if q < 15:
            continue
        else:
            gdelt_success = False
            break

if gdelt_success == True:   
    soup = bs(r2.content, 'html.parser')
    s = soup.find_all('script')
    item = [x for x in s if 'xaxiscats' in str(x) and 'Highcharts.setOptions' in str(x)][0]
    
    x = str(item).split('Date.UTC')[1:-1]
    x = [dummy[1:5]+'-'+dummy[7:9]+'-'+dummy[13:15] for dummy in x]
    
    y = str(item).split('{y:')[1:-1]
    y = [dummy[1:-3] for dummy in y]
    
    gdelt = {k:float(v) for k,v in zip(x,y)}
    gdelt = pd.DataFrame.from_dict({'News':gdelt})
    
    gdelt = gdelt.reset_index(drop=False)
else:
    gdelt = pd.read_csv('https://acadic-portal.github.io/Avian/dummies/gdelt.csv')
    
gdelt.to_csv('gdelt.csv', index=False)

#%%
# Training
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#import tensorflow as tf
import datetime

from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
#from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import math
from tqdm import tqdm
import scipy.stats as stt
from scipy import optimize
import os
import shutil

#%%
# Read data
cases = pd.read_csv('cases.csv')
gt = pd.read_csv('gt.csv')
gn = pd.read_csv('gn1.csv')
reddit = pd.read_csv('reddit2.csv')
gdelt = pd.read_csv('gdelt.csv')

end = datetime.datetime.now() #- datetime.timedelta(days=1)
start = end - datetime.timedelta(days= 600)
end = end.strftime('%Y-%m-%d')
start = start.strftime('%Y-%m-%d')
start1 = start
complete_dates = [start1]
while pd.to_datetime(start1) < pd.to_datetime(end):
    start1 = datetime.datetime.strptime(start1, '%Y-%m-%d') + datetime.timedelta(days= 1)
    start1 = start1.strftime('%Y-%m-%d')
    complete_dates.append(start1)
    
cases['date'] = [x[:10] for x in cases['Collection Date']]
reddit['Date'] = [x[:10] for x in reddit['date']]
gt = gt.set_index('date')
gdelt = gdelt.set_index('index')

#%%
# Prepare country- and regional-level data.
ca = {}
w = {}
c = {}
a = {}
ca['cases'] = {dt:len(cases[cases['date'] == dt]) for dt in complete_dates}
data = cases[cases['Province'].str.contains('BC|AB|SK|MB', na= False)]
w['cases'] = {dt:len(data[data['date'] == dt]) for dt in complete_dates}
data = cases[cases['Province'].str.contains('ON|QC', na= False)]
c['cases'] = {dt:len(data[data['date'] == dt]) for dt in complete_dates}
data = cases[cases['Province'].str.contains('NS|NB|NL|PE', na= False)]
a['cases'] = {dt:len(cases[cases['date'] == dt]) for dt in complete_dates}
if gn_success == True:
    ca['gn'] = {dt:len(gn[gn['Date'] == dt]) for dt in complete_dates}
    w['gn'] = {dt:len(gn[gn['Date'] == dt]) for dt in complete_dates}
    c['gn'] = {dt:len(gn[gn['Date'] == dt]) for dt in complete_dates}
    a['gn'] = {dt:len(gn[gn['Date'] == dt]) for dt in complete_dates}
if r_success == True:
    ca['r'] = {dt:len(reddit[reddit['Date'] == dt]) for dt in complete_dates}
    data = reddit[reddit['subreddit'].str.lower().str.contains('alberta|calgary|edmonton|britishcolumbia|vancouver|regina|saskatchewan|manitoba|winnipeg|vancouvercanada|british_columbia|winnipegnews|AlbertaNews|VancouverNews', na=False)]
    w['r'] = {dt:len(data[data['Date'] == dt]) for dt in complete_dates}
    data = reddit[reddit['subreddit'].str.lower().str.contains('ontario|kingstonontario|toronto|ottawa|quebec|montreal|ottawanews|ontarionews|torontonews|quebecnews', na=False)]
    c['r'] = {dt:len(data[data['Date'] == dt]) for dt in complete_dates}
    data = reddit[reddit['subreddit'].str.lower().str.contains('novascotia|novascotiagardening|halifax|newbrunswickcanada|newbrunswick|newfoundland|newfoundland_labrador|princeedwardisland|princeedwardcounty|charlottetown|atlanticcanada|halifaxnews', na=False)]
    a['r'] = {dt:len(data[data['Date'] == dt]) for dt in complete_dates}
if gt_success == True:
    ca['gt'] = {dt:gt['/m/0292d3'][dt] if dt in list(gt.index) and pd.isnull(gt['/m/0292d3'][dt]) == False else 0.01 for dt in complete_dates}
    w['gt'] = {dt:gt['/m/0292d3'][dt] if dt in list(gt.index) and pd.isnull(gt['/m/0292d3'][dt]) == False else 0.01 for dt in complete_dates}
    c['gt'] = {dt:gt['/m/0292d3'][dt] if dt in list(gt.index) and pd.isnull(gt['/m/0292d3'][dt]) == False else 0.01 for dt in complete_dates}
    a['gt'] = {dt:gt['/m/0292d3'][dt] if dt in list(gt.index) and pd.isnull(gt['/m/0292d3'][dt]) == False else 0.01 for dt in complete_dates}
if gdelt_success == True:
    ca['gdelt'] = {dt:gdelt['News'][dt] if dt in list(gdelt.index) and pd.isnull(gdelt['News'][dt]) == False else 0.01 for dt in complete_dates}
    w['gdelt'] = {dt:gdelt['News'][dt] if dt in list(gdelt.index) and pd.isnull(gdelt['News'][dt]) == False else 0.01 for dt in complete_dates}
    c['gdelt'] = {dt:gdelt['News'][dt] if dt in list(gdelt.index) and pd.isnull(gdelt['News'][dt]) == False else 0.01 for dt in complete_dates}
    a['gdelt'] = {dt:gdelt['News'][dt] if dt in list(gdelt.index) and pd.isnull(gdelt['News'][dt]) == False else 0.01 for dt in complete_dates}
ca = pd.DataFrame.from_dict(ca)
w = pd.DataFrame.from_dict(w)
c = pd.DataFrame.from_dict(c)
a = pd.DataFrame.from_dict(a)

#%%
# Center and scale
ca1 = ca[ca.index.isin(complete_dates[-584:])]
mmscaler_ca = MinMaxScaler()
ca1 = ca1.fillna(0)
#w1 = w1.drop (columns= ['wahis','domestic','rain','prec'])
ca1 = mmscaler_ca.fit_transform (ca1)

w1 = w[w.index.isin(complete_dates[-349:])]
mmscaler_w = MinMaxScaler()
w1 = w1.fillna(0)
#w1 = w1.drop (columns= ['wahis','domestic','rain','prec'])
w1 = mmscaler_w.fit_transform (w1)

c1 = c[c.index.isin(complete_dates[-484:])]
mmscaler_c = MinMaxScaler()
c1 = c1.fillna(0)
#c1 = c1.drop (columns= ['wahis','domestic','rain','prec'])
c1 = mmscaler_c.fit_transform (c1)

a1 = a[a.index.isin(complete_dates[-484:])]
mmscaler_a = MinMaxScaler()
a1 = a1.fillna(0)
#a1 = a1.drop (columns= ['wahis','domestic','rain','prec'])
a1 = mmscaler_a.fit_transform (a1)

#%%
# Model
class myDataset (Dataset):
  def __init__(self, sequences):
    self.sequences = sequences

  def __len__ (self):
    return len (self.sequences)

  def __getitem__(self, idx):
    sequence, label = self.sequences[idx]
    return dict (sequence= torch.Tensor(sequence), label = torch.Tensor(label)) # The difference between Tensor and tensor is that Tensor is always float, but tensor is the actual type of the data.

#-----------------------------------------------------------

class myDataModule (pl.LightningDataModule):
  def __init__(self, train_sequences, test_sequences, batch_size):
    super().__init__()
    self.train_sequences = train_sequences
    self.batch_size = batch_size
    self.test_sequences = test_sequences

  def setup (self, stage=None):
    self.train_dataset = myDataset (self.train_sequences)
    self.test_dataset = myDataset (self.test_sequences)

  def train_dataloader (self):
    return DataLoader (self.train_dataset, batch_size= self.batch_size, shuffle= False, num_workers=2)

  def val_dataloader (self):
    return DataLoader (self.test_dataset, batch_size=1, shuffle=False, num_workers= 1)

  def test_dataloader (self):
    return DataLoader (self.test_dataset, batch_size= 1, shuffle=False, num_workers= 1)

#-----------------------------------------------------------

class ewsf1 (nn.Module):
  def __init__ (self, n_features, n_hidden, dropout, n_layers, seq_length, step_ahead):
    super().__init__()
    self.gru = nn.GRU (input_size= n_features, hidden_size= n_hidden, batch_first=True, num_layers=n_layers, dropout=dropout)#, bidirectional= True)
#    self.lstm = nn.LSTM (input_size= n_features, hidden_size= n_hidden, batch_first=True, num_layers=n_layers, dropout= dropout)

    self.regressor = nn.Linear (n_hidden, step_ahead)

  def forward (self, x):
    self.gru.flatten_parameters()
    _, hidden = self.gru (x)
    out = hidden[-1]
    out = self.regressor(out)

#    self.lstm.flatten_parameters()
#    _, (hidden, _) = self.lstm (x)
#    out = hidden[-1]
#    out = self.regressor(out)

    return out

#-----------------------------------------------------------

class ewsf1Module (pl.LightningModule):
  def __init__(self, n_features: int, learning_rate, optimizerr, n_hidden, n_layers, dropout, seq_length, step_ahead):
  #def __init__(self, n_features: int, learning_rate, optimizerr, n_hidden, n_layers, dropout1, dropout2, kernel, seq_length, step_ahead):
    super().__init__()
    self.model = ewsf1 (n_features, n_hidden, dropout, n_layers, seq_length, step_ahead)
    #self.model = ewsf1 (n_features, n_hidden, dropout1, dropout2, n_layers, seq_length, kernel, step_ahead)
    self.criterion = nn.MSELoss()
    self.lr = learning_rate
    self.optimizerr = optimizerr
    self.r2 = -5
    self.mape = 5

  def forward (self, x, labels= None):
    output = self.model (x)
    loss = 0
    if labels is not None:
      loss = self.criterion (output, labels)#.unsqueeze(dim=1))
    return loss, output

  def training_step (self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]

    loss, outputs = self (sequences, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss":loss}

  def validation_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self (sequences, labels)
    #print (labels.shape, outputs.shape)
    a = []
    p = []
    for i in range (len(labels[0])):
      a.append (labels[0][i].item())
      p.append (outputs[0][i].item())
    r2 = r2_score(a,p)
    if r2 > 1:
      r2 = -r2
    if r2 > self.r2:
      self.r2 = r2
    self.log ("R2", r2, prog_bar= True, logger= True)
    self.log ("val_loss", loss, prog_bar= True, logger= True)

    mape = [abs(labels[0][i].item() - outputs[0][i].item()) / abs(labels[0][i].item()) for i in range(len(labels[0])) if labels[0][i].item() != 0]
    mape = np.sum(mape) / len(mape)
    if mape < self.mape:
      self.mape = mape
    self.log ("MAPE", mape, prog_bar= True, logger= True)

    rmse = [(labels[0][i].item() - outputs[0][i].item())*(labels[0][i].item() - outputs[0][i].item()) for i in range(len(labels[0])) if labels[0][i].item() != 0]
    rmse = np.sum(rmse) / len(rmse)
    rmse = math.sqrt(rmse)
    self.log ("RMSE", rmse, prog_bar= True, logger= True)

    #wandb.log ({"val_loss": loss, "R2": self.r2, 'MAPE':self.mape, 'RMSE':rmse})
    return loss

  def test_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self (sequences, labels)
    self.log ("test_loss", loss, prog_bar= True, logger= True)
    return loss

  def configure_optimizers (self):
    if self.optimizerr == "sgd":
      optimizer = optim.SGD(self.parameters(), lr= self.lr, momentum=0.9)
    elif self.optimizerr == "adam":
      optimizer = optim.Adam(self.parameters(), lr= self.lr)
    else:
      optimizer = optim.AdamW (self.parameters(), lr= self.lr)
    return optimizer

#%%
checkpoint_path = os.path.join("checkpoints", "best-checkpoint.ckpt")

#%%
# Train Canada
p = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
#a = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
train_length = 400
seq_size = 150
step_ahead = 14
step = 20

#do1 = 0.1
do = 0.5
batch_size = 32
hidden = 128
lr = 0.0001
layers = 2
opt = "AdamW"
rseed = 95
kernel = 2
epochs = 1000

train_sequences = []
test_sequences = []

sequence = []
#label = []

for j in range (train_length+seq_size, len(ca1)-step_ahead, step):
  train_dummy = []
  test_dummy = []

  for i in range (j-train_length, j): # This transpose here has been added for cnn
    #train_dummy.append ((np.transpose (data1[i-seq_size:i, :]), data1[i:i+step_ahead, 0])) # The output is at column 0
    train_dummy.append (( (ca1[i-seq_size:i, :]), ca1[i:i+step_ahead, 0])) # The output is at column 0
  for i in range (j, j+step):
    if i < len(ca1)-step_ahead:
      #test_dummy.append ((np.transpose (data1[i-seq_size:i, :]), data1[i:i+step_ahead, 0]))
      test_dummy.append (( (ca1[i-seq_size:i, :]), ca1[i:i+step_ahead, 0]))
    else:
      #test_dummy.append ((np.transpose (data1[i-seq_size:i, :]), data1[i:, 0]))
      test_dummy.append (( (ca1[i-seq_size:i, :]), ca1[i:, 0]))
      break

  train_sequences.append(train_dummy)
  test_sequences.append(test_dummy)

# Building for prediction

#  l = []
i = len(ca1)
s = torch.Tensor ( (ca1[i-seq_size:i, :]))
sequence = [s]

#--------------------------------------------------------------

#n_features = train_sequences[0][0][0].shape[1]
n_features = ca1.shape[1]

#--------------------------------------------------------------

p = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
#a = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
train_length = 400
seq_size = 150
step_ahead = 14
step = 20

#do1 = 0.1
do = 0.5
batch_size = 32
hidden = 128
lr = 0.0001
layers = 2
opt = "AdamW"
rseed = 95
#kernel = 2
epochs = 1000

for i in range(len(train_sequences)):
  trn = train_sequences[i]
  tst = test_sequences[i]

  
  #if os.path.exists('/content/checkpoints'):
  #  shutil.rmtree('/content/checkpoints')
  if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)

  pl.seed_everything (rseed)

  dm = myDataModule (trn, tst, batch_size)
  dm.setup()

  checkpoint_callback = ModelCheckpoint (dirpath="checkpoints", filename="best-checkpoint", save_top_k=1, verbose=False, monitor="val_loss", mode="min")
  logger = TensorBoardLogger ("lightning_logs", name="ewsf")
  early_stopping_callback = EarlyStopping (monitor= "val_loss", mode="min", patience= 100) #(monitor= "val_loss", patience= 2)
  trainer = pl.Trainer ( logger= logger, callbacks=[early_stopping_callback, checkpoint_callback], max_epochs= epochs)#, gpus=1)#, progress_bar_refresh_rate=30) #devices=1, accelerator="gpu")

  model = ewsf1Module (n_features= n_features, learning_rate=lr, optimizerr=opt, n_hidden=hidden, n_layers=layers, dropout=do, seq_length=seq_size, step_ahead=step_ahead)
  #model = ewsf1Module (n_features= n_features, learning_rate= lr, optimizerr= opt, n_hidden= hidden, n_layers= layers, dropout1= do1, dropout2= do2, kernel= kernel, seq_length= seq_size, step_ahead= step_ahead)
  #trainer.tune (model, dm)

  trainer.fit (model, dm)

#  s = sequence[0]
#  l = label[i]

#  for j in range(len(sequence[i])):
#    s = sequence[i][j]
#    l = label[i][j]
#  #trained_model = trainer.model
#    trained_model = ewsf1Module.load_from_checkpoint ("/content/checkpoints/best-checkpoint.ckpt", n_features= n_features, learning_rate=lr, optimizerr=opt, n_hidden=hidden, n_layers=layers, dropout=do, seq_length=seq_size, step_ahead=step_ahead)
#    #trained_model = ewsf1Module.load_from_checkpoint ("/content/checkpoints/best-checkpoint.ckpt", n_features= n_features, learning_rate= lr, optimizerr= opt, n_hidden= hidden, n_layers= layers, dropout1= do1, dropout2= do2, kernel= kernel, seq_length= seq_size, step_ahead= step_ahead)

#    _, output = trained_model (s.unsqueeze (dim=0))
#    #break
#    for k in range(output.squeeze().shape[0]):
#      p[k+1].append (output.squeeze()[k].item())
#      a[k+1].append (l.squeeze()[k].item())
#    #print (output, label)

#  p_dummy = pd.DataFrame.from_records(p)
#  a_dummy = pd.DataFrame.from_records(a)
#  p_dummy.to_csv ('/content/drive/MyDrive/EWSF/Dashboard/H5N1/files/p_result_ca.csv', index= False)
#  a_dummy.to_csv ('/content/drive/MyDrive/EWSF/Dashboard/H5N1/files/a_result_ca.csv', index = False)
#  print(i, 'completed:-------------------------------------------------------------------------------------------------------------------------------->', len(p_dummy), len(a_dummy))
#  #break

#-------------------------------------------------------------

# Predict
p = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
trained_model = ewsf1Module.load_from_checkpoint (checkpoint_path, n_features= n_features, learning_rate=lr, optimizerr=opt, n_hidden=hidden, n_layers=layers, dropout=do, seq_length=seq_size, step_ahead=step_ahead)
s = sequence[0]
_, output = trained_model (s.unsqueeze (dim=0))
for k in range(output.squeeze().shape[0]):
      p[k+1].append (output.squeeze()[k].item())
p_dummy = pd.DataFrame.from_records(p)
p_dummy.to_csv ('p_result_ca.csv', index= False)

scale1 = MinMaxScaler()
scale1.min_, scale1.scale_ = mmscaler_ca.min_[0], mmscaler_ca.scale_[0]
dummy = scale1.inverse_transform(np.array(p_dummy).transpose())
p_dummy_scaled = pd.DataFrame(data=dummy.transpose(), columns=list(range(1,15)))
p_dummy.to_csv ('p_result_ca_scaled.csv', index= False)

#%%
# Train Western
p = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
#a = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
train_length = 300
seq_size = 15
step_ahead = 14
step = 20

#do1 = 0.1
#do2 = 0.5
do = 0.5
batch_size = 8
hidden = 128
lr = 0.0001
layers = 2
opt = "Adam"
rseed = 60
#kernel = 2
epochs = 1000

train_sequences = []
test_sequences = []

sequence = []
label = []

# data1_w:
train_data = w1.copy()

for j in range (train_length+seq_size, len(train_data)-step_ahead, step):
  train_dummy = []
  test_dummy = []

  for i in range (j-train_length, j): # This transpose here has been added for cnn
    #train_dummy.append ((np.transpose (data1[i-seq_size:i, :]), data1[i:i+step_ahead, 0])) # The output is at column 0
    train_dummy.append ((train_data[i-seq_size:i, :], train_data[i:i+step_ahead, 0])) # The output is at column 0
  for i in range (j, j+step):
    if i < len(train_data)-step_ahead:
      #test_dummy.append ((np.transpose (data1[i-seq_size:i, :]), data1[i:i+step_ahead, 0]))
      test_dummy.append ( (train_data[i-seq_size:i, :], train_data[i:i+step_ahead, 0]))
    else:
      test_dummy.append ((train_data[i-seq_size:i, :], train_data[i:, 0]))
      break

  train_sequences.append(train_dummy)
  test_sequences.append(test_dummy)

i = len(w1)
s = torch.Tensor (train_data[i-seq_size:i, :])
sequence = [s]

#---------------------------------------------------------------

#n_features = train_sequences[0][0][0].shape[1]
n_features = train_data.shape[1]

#---------------------------------------------------------------

p = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
#a = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
train_length = 300
seq_size = 15
step_ahead = 14
step = 20

#do1 = 0.1
#do2 = 0.5
do = 0.5
batch_size = 8
hidden = 128
lr = 0.0001
layers = 2
opt = "Adam"
rseed = 60
#kernel = 2
epochs = 1000

#for i in range(13,15):
for i in range(len(train_sequences)):
  trn = train_sequences[i]
  tst = test_sequences[i]

  #if os.path.exists('/content/checkpoints'):
  #  shutil.rmtree('/content/checkpoints')
  if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)

  pl.seed_everything (rseed)

  dm = myDataModule (trn, tst, batch_size)
  dm.setup()

  checkpoint_callback = ModelCheckpoint (dirpath="checkpoints", filename="best-checkpoint", save_top_k=1, verbose=False, monitor="val_loss", mode="min")
  logger = TensorBoardLogger ("lightning_logs", name="ewsf")
  early_stopping_callback = EarlyStopping (monitor= "val_loss", mode="min", patience= 100) #(monitor= "val_loss", patience= 2)
  trainer = pl.Trainer ( logger= logger, callbacks=[early_stopping_callback, checkpoint_callback], max_epochs= epochs)#, gpus=1)#, progress_bar_refresh_rate=30) #devices=1, accelerator="gpu")

  model = ewsf1Module (n_features= n_features, learning_rate=lr, optimizerr=opt, n_hidden=hidden, n_layers=layers, dropout=do, seq_length=seq_size, step_ahead=step_ahead)
  #model = ewsf1Module (n_features= n_features, learning_rate= lr, optimizerr= opt, n_hidden= hidden, n_layers= layers, dropout1= do1, dropout2= do2, kernel= kernel, seq_length= seq_size, step_ahead= step_ahead)
  #trainer.tune (model, dm)

  trainer.fit (model, dm)

#---------------------------------------------------------------

# Predict
p = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
trained_model = ewsf1Module.load_from_checkpoint (checkpoint_path, n_features= n_features, learning_rate=lr, optimizerr=opt, n_hidden=hidden, n_layers=layers, dropout=do, seq_length=seq_size, step_ahead=step_ahead)
s = sequence[0]
_, output = trained_model (s.unsqueeze (dim=0))
for k in range(output.squeeze().shape[0]):
      p[k+1].append (output.squeeze()[k].item())
p_dummy = pd.DataFrame.from_records(p)
p_dummy.to_csv ('p_result_w.csv', index= False)

scale1 = MinMaxScaler()
scale1.min_, scale1.scale_ = mmscaler_w.min_[0], mmscaler_w.scale_[0]
dummy = scale1.inverse_transform(np.array(p_dummy).transpose())
p_dummy_scaled = pd.DataFrame(data=dummy.transpose(), columns=list(range(1,15)))
p_dummy.to_csv ('p_result_w_scaled.csv', index= False)

#%%
# Train Central
p = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
#a = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
train_length = 300
seq_size = 150
step_ahead = 14
step = 20

#do1 = 0.1
#do2 = 0.5
do = 0.1
batch_size = 32
hidden = 256
lr = 0.001
layers = 5
opt = "AdamW"
rseed = 50
#kernel = 2
epochs = 1000

train_sequences = []
test_sequences = []

sequence = []
label = []

# data1_c:
train_data = c1.copy()

for j in range (train_length+seq_size, len(train_data)-step_ahead, step):
  train_dummy = []
  test_dummy = []

  for i in range (j-train_length, j): # This transpose here has been added for cnn
    #train_dummy.append ((np.transpose (data1[i-seq_size:i, :]), data1[i:i+step_ahead, 0])) # The output is at column 0
    train_dummy.append ((train_data[i-seq_size:i, :], train_data[i:i+step_ahead, 0])) # The output is at column 0
  for i in range (j, j+step):
    if i < len(train_data)-step_ahead:
      #test_dummy.append ((np.transpose (data1[i-seq_size:i, :]), data1[i:i+step_ahead, 0]))
      test_dummy.append ( (train_data[i-seq_size:i, :], train_data[i:i+step_ahead, 0]))
    else:
      test_dummy.append ((train_data[i-seq_size:i, :], train_data[i:, 0]))
      break

  train_sequences.append(train_dummy)
  test_sequences.append(test_dummy)

i = len(c1)
s = torch.Tensor(train_data[i-seq_size:i, :])
sequence = [s]

#------------------------------------------------------------

#n_features = train_sequences[0][0][0].shape[1]
n_features = train_data.shape[1]

#------------------------------------------------------------

p = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
#a = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
train_length = 300
seq_size = 150
step_ahead = 14
step = 20

#do1 = 0.1
#do2 = 0.5
do = 0.1
batch_size = 32
hidden = 256
lr = 0.001
layers = 5
opt = "AdamW"
rseed = 50
#kernel = 2
epochs = 1000

for i in range(len(train_sequences)):
  trn = train_sequences[i]
  tst = test_sequences[i]

  #if os.path.exists('/content/checkpoints'):
  #  shutil.rmtree('/content/checkpoints')
  if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)

  pl.seed_everything (rseed)

  dm = myDataModule (trn, tst, batch_size)
  dm.setup()

  checkpoint_callback = ModelCheckpoint (dirpath="checkpoints", filename="best-checkpoint", save_top_k=1, verbose=False, monitor="val_loss", mode="min")
  logger = TensorBoardLogger ("lightning_logs", name="ewsf")
  early_stopping_callback = EarlyStopping (monitor= "val_loss", mode="min", patience= 100) #(monitor= "val_loss", patience= 2)
  trainer = pl.Trainer ( logger= logger, callbacks=[early_stopping_callback, checkpoint_callback], max_epochs= epochs)#, gpus=1)#, progress_bar_refresh_rate=30) #devices=1, accelerator="gpu")

  model = ewsf1Module (n_features= n_features, learning_rate=lr, optimizerr=opt, n_hidden=hidden, n_layers=layers, dropout=do, seq_length=seq_size, step_ahead=step_ahead)
  #model = ewsf1Module (n_features= n_features, learning_rate= lr, optimizerr= opt, n_hidden= hidden, n_layers= layers, dropout1= do1, dropout2= do2, kernel= kernel, seq_length= seq_size, step_ahead= step_ahead)
  #trainer.tune (model, dm)

  trainer.fit (model, dm)

#------------------------------------------------------------

# Predict
p = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
trained_model = ewsf1Module.load_from_checkpoint (checkpoint_path, n_features= n_features, learning_rate=lr, optimizerr=opt, n_hidden=hidden, n_layers=layers, dropout=do, seq_length=seq_size, step_ahead=step_ahead)
s = sequence[0]
_, output = trained_model (s.unsqueeze (dim=0))
for k in range(output.squeeze().shape[0]):
      p[k+1].append (output.squeeze()[k].item())
p_dummy = pd.DataFrame.from_records(p)
p_dummy.to_csv ('p_result_c.csv', index= False)

scale1 = MinMaxScaler()
scale1.min_, scale1.scale_ = mmscaler_c.min_[0], mmscaler_c.scale_[0]
dummy = scale1.inverse_transform(np.array(p_dummy).transpose())
p_dummy_scaled = pd.DataFrame(data=dummy.transpose(), columns=list(range(1,15)))
p_dummy.to_csv ('p_result_c_scaled.csv', index= False)

#%%
p = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
#a = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
train_length = 300
seq_size = 150
step_ahead = 14
step = 20

#do1 = 0.1
#do2 = 0.5
do = 0.5
batch_size = 32
hidden = 128
lr = 0.0001
layers = 2
opt = "AdamW"
rseed = 30
#kernel = 2
epochs = 1000

train_sequences = []
test_sequences = []

sequence = []
label = []

# data1_a:
train_data = a1.copy()

for j in range (train_length+seq_size, len(train_data)-step_ahead, step):
  train_dummy = []
  test_dummy = []

  for i in range (j-train_length, j): # This transpose here has been added for cnn
    #train_dummy.append ((np.transpose (data1[i-seq_size:i, :]), data1[i:i+step_ahead, 0])) # The output is at column 0
    train_dummy.append ((train_data[i-seq_size:i, :], train_data[i:i+step_ahead, 0])) # The output is at column 0
  for i in range (j, j+step):
    if i < len(train_data)-step_ahead:
      #test_dummy.append ((np.transpose (data1[i-seq_size:i, :]), data1[i:i+step_ahead, 0]))
      test_dummy.append ( (train_data[i-seq_size:i, :], train_data[i:i+step_ahead, 0]))
    else:
      test_dummy.append ((train_data[i-seq_size:i, :], train_data[i:, 0]))
      break

  train_sequences.append(train_dummy)
  test_sequences.append(test_dummy)

i = len(a1)
s = torch.Tensor (train_data[i-seq_size:i, :])
sequence = [s]

#------------------------------------------------------------

#n_features = train_sequences[0][0][0].shape[1]
n_features = train_data.shape[1]

#------------------------------------------------------------

p = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
#a = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
train_length = 300
seq_size = 150
step_ahead = 14
step = 20

#do1 = 0.1
#do2 = 0.5
do = 0.5
batch_size = 32
hidden = 128
lr = 0.0001
layers = 2
opt = "AdamW"
rseed = 30
#kernel = 2
epochs = 1000

for i in range(len(train_sequences)):
  trn = train_sequences[i]
  tst = test_sequences[i]

  #if os.path.exists('/content/checkpoints'):
  #  shutil.rmtree('/content/checkpoints')
  if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)

  pl.seed_everything (rseed)

  dm = myDataModule (trn, tst, batch_size)
  dm.setup()

  checkpoint_callback = ModelCheckpoint (dirpath="checkpoints", filename="best-checkpoint", save_top_k=1, verbose=False, monitor="val_loss", mode="min")
  logger = TensorBoardLogger ("lightning_logs", name="ewsf")
  early_stopping_callback = EarlyStopping (monitor= "val_loss", mode="min", patience= 100) #(monitor= "val_loss", patience= 2)
  trainer = pl.Trainer ( logger= logger, callbacks=[early_stopping_callback, checkpoint_callback], max_epochs= epochs)#, gpus=1)#, progress_bar_refresh_rate=30) #devices=1, accelerator="gpu")

  model = ewsf1Module (n_features= n_features, learning_rate=lr, optimizerr=opt, n_hidden=hidden, n_layers=layers, dropout=do, seq_length=seq_size, step_ahead=step_ahead)
  #model = ewsf1Module (n_features= n_features, learning_rate= lr, optimizerr= opt, n_hidden= hidden, n_layers= layers, dropout1= do1, dropout2= do2, kernel= kernel, seq_length= seq_size, step_ahead= step_ahead)
  #trainer.tune (model, dm)

  trainer.fit (model, dm)

#------------------------------------------------------------

# Predict
p = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
trained_model = ewsf1Module.load_from_checkpoint (checkpoint_path, n_features= n_features, learning_rate=lr, optimizerr=opt, n_hidden=hidden, n_layers=layers, dropout=do, seq_length=seq_size, step_ahead=step_ahead)
s = sequence[0]
_, output = trained_model (s.unsqueeze (dim=0))
for k in range(output.squeeze().shape[0]):
      p[k+1].append (output.squeeze()[k].item())
p_dummy = pd.DataFrame.from_records(p)
p_dummy.to_csv ('p_result_a.csv', index= False)

scale1 = MinMaxScaler()
scale1.min_, scale1.scale_ = mmscaler_a.min_[0], mmscaler_a.scale_[0]
dummy = scale1.inverse_transform(np.array(p_dummy).transpose())
p_dummy_scaled = pd.DataFrame(data=dummy.transpose(), columns=list(range(1,15)))
p_dummy.to_csv ('p_result_a_scaled.csv', index= False)

#%%
# Build Map
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import scipy.stats as stt
import math
import os
from tqdm import tqdm
from scipy import optimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
#import pycountry
import json
import geopandas as gpd
import json

#%%
# Read Map
cn = gpd.read_file ('https://acadic-portal.github.io/Avian/regional/Canada_regions.shp')
cn = cn.to_crs(4326)

cn = cn.drop(columns= ['OBJECTID_1','PRUID','PRNAME','PRFNAME','PREABBR','PRFABBR'])
code = {'Alberta':'WR', 'Ontario':'CR', 'New Brunswick':'AR', 'Yukon':'YT', 'Northwest Territories':'NT', 'Nunavut':'NU'}
cn['code'] = [code[x] for x in list(cn['PRENAME'])]
prename = {'Alberta':'Western Region', 'Ontario':'Central Region', 'New Brunswick':'Atlantic Region', 'Yukon':'Yukon', 'Northwest Territories':'Northwest Territories', 'Nunavut':'Nunavut'}
cn['PRENAME'] = [prename[x] for x in list(cn['PRENAME'])]

geometry = cn['geometry']
cn = cn.drop(columns= ['geometry'])
cn = gpd.GeoDataFrame(cn, crs="EPSG:4326", geometry=geometry)
cn.to_file('Canada_regional_raw.json', encoding= 'utf-8', driver="GeoJSON")

#%%
# Make sure of the dates
end = datetime.datetime.now()
start = end - datetime.timedelta(days= 600)
end = end.strftime('%Y-%m-%d')
start = start.strftime('%Y-%m-%d')

start1 = start
complete_dates = [start1]
while pd.to_datetime(start1) < pd.to_datetime(end):
    start1 = datetime.datetime.strptime(start1, '%Y-%m-%d') + datetime.timedelta(days= 1)
    start1 = start1.strftime('%Y-%m-%d')
    complete_dates.append(start1)
    
#%%
dummy = pd.read_csv('p_result_ca_scaled.csv')#.drop(columns='Unnamed: 0')
ca = {}
ca['cases'] = {dt:len(cases[cases['date'] == dt]) for dt in complete_dates}
ca['gn'] = {dt:len(gn[gn['Date'] == dt]) for dt in complete_dates}
ca['r'] = {dt:len(reddit[reddit['Date'] == dt]) for dt in complete_dates}
ca['gt'] = {dt:gt['/m/0292d3'][dt] if dt in list(gt.index) and pd.isnull(gt['/m/0292d3'][dt]) == False else 0.01 for dt in complete_dates}
ca['gdelt'] = {dt:gdelt['News'][dt] if dt in list(gdelt.index) and pd.isnull(gdelt['News'][dt]) == False else 0.01 for dt in complete_dates}
ca = pd.DataFrame.from_dict(ca)
ca['Predictions'] = np.abs(np.array(dummy)[0,:]).tolist() + [np.nan]*(len(ca)-14)

ca = ca.reset_index()
ca.to_json('avian_country_canada.json', index=False)

#%%
w = {}
data = cases[cases['Province'].str.contains('BC|AB|SK|MB', na= False)]
w['cases'] = {dt:len(data[data['date'] == dt]) for dt in complete_dates}
w['gn'] = {dt:len(gn[gn['Date'] == dt]) for dt in complete_dates}
data = reddit[reddit['subreddit'].str.lower().str.contains('alberta|calgary|edmonton|britishcolumbia|vancouver|regina|saskatchewan|manitoba|winnipeg|vancouvercanada|british_columbia|winnipegnews|AlbertaNews|VancouverNews', na=False)]
w['r'] = {dt:len(data[data['Date'] == dt]) for dt in complete_dates}
w['gt'] = {dt:gt['/m/0292d3'][dt] if dt in list(gt.index) and pd.isnull(gt['/m/0292d3'][dt]) == False else 0.01 for dt in complete_dates}
w['gdelt'] = {dt:gdelt['News'][dt] if dt in list(gdelt.index) and pd.isnull(gdelt['News'][dt]) == False else 0.01 for dt in complete_dates}
w = pd.DataFrame.from_dict(w)

w = w.reset_index()
w.to_json('avian_western_canada.json', index=False)#, encoding= 'utf-8')

#%%
c = {}
data = cases[cases['Province'].str.contains('ON|QC', na= False)]
c['cases'] = {dt:len(data[data['date'] == dt]) for dt in complete_dates}
c['gn'] = {dt:len(gn[gn['Date'] == dt]) for dt in complete_dates}
data = reddit[reddit['subreddit'].str.lower().str.contains('ontario|kingstonontario|toronto|ottawa|quebec|montreal|ottawanews|ontarionews|torontonews|quebecnews', na=False)]
c['r'] = {dt:len(data[data['Date'] == dt]) for dt in complete_dates}
c['gt'] = {dt:gt['/m/0292d3'][dt] if dt in list(gt.index) and pd.isnull(gt['/m/0292d3'][dt]) == False else 0.01 for dt in complete_dates}
c['gdelt'] = {dt:gdelt['News'][dt] if dt in list(gdelt.index) and pd.isnull(gdelt['News'][dt]) == False else 0.01 for dt in complete_dates}
c = pd.DataFrame.from_dict(c)

c = c.reset_index()
c.to_json('avian_central_canada.json', index=False)#, encoding= 'utf-8')

#%%
a = {}
data = cases[cases['Province'].str.contains('NS|NB|NL|PE', na= False)]
a['cases'] = {dt:len(cases[cases['date'] == dt]) for dt in complete_dates}
a['gn'] = {dt:len(gn[gn['Date'] == dt]) for dt in complete_dates}
data = reddit[reddit['subreddit'].str.lower().str.contains('novascotia|novascotiagardening|halifax|newbrunswickcanada|newbrunswick|newfoundland|newfoundland_labrador|princeedwardisland|princeedwardcounty|charlottetown|atlanticcanada|halifaxnews', na=False)]
a['r'] = {dt:len(data[data['Date'] == dt]) for dt in complete_dates}
a['gt'] = {dt:gt['/m/0292d3'][dt] if dt in list(gt.index) and pd.isnull(gt['/m/0292d3'][dt]) == False else 0.01 for dt in complete_dates}
a['gdelt'] = {dt:gdelt['News'][dt] if dt in list(gdelt.index) and pd.isnull(gdelt['News'][dt]) == False else 0.01 for dt in complete_dates}
a = pd.DataFrame.from_dict(a)

a = a.reset_index()
a.to_json('avian_atlantic_canada.json', index=False)#, encoding= 'utf-8')

#%%
with open("Canada_regional_raw.json", "r") as read_file:
    data = json.load(read_file)

result = {'WR':w.reset_index(),'CR':c.reset_index(),'AR':a.reset_index()}
for i in tqdm(range(len(data['features']))):
    item = data['features'][i]
    iso = item['properties']['code']
    if iso in list(result.keys()):
        for c in result[iso].columns:
            data['features'][i]['properties'][c] = list(result[iso][c])
            
#%%
predicted = {'WR': pd.read_csv('p_result_w_scaled.csv'),#.drop(columns='Unnamed: 0'),
             'CR': pd.read_csv('p_result_c_scaled.csv'),#.drop(columns='Unnamed: 0'),
             'AR': pd.read_csv('p_result_a_scaled.csv'),#.drop(columns='Unnamed: 0'),
}

predicted['WR'].columns = [(datetime.datetime.strptime(end, '%Y-%m-%d') + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1,15)]
predicted['CR'].columns = [(datetime.datetime.strptime(end, '%Y-%m-%d') + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1,15)]
predicted['AR'].columns = [(datetime.datetime.strptime(end, '%Y-%m-%d') + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1,15)]

predicted['WR'] = predicted['WR'].to_dict()
predicted['CR'] = predicted['CR'].to_dict()
predicted['AR'] = predicted['AR'].to_dict()

for k in predicted['WR'].keys():
  predicted['WR'][k][0] = round(np.abs(10 * predicted['WR'][k][0]), 4)
  predicted['CR'][k][0] = round(np.abs(10 * predicted['CR'][k][0]), 4)
  predicted['AR'][k][0] = round(np.abs(10 * predicted['AR'][k][0]), 4)

for i in tqdm(range(len(data['features']))):
    item = data['features'][i]
    iso = item['properties']['code']
    if iso in list(predicted.keys()):
      for k in predicted[iso].keys():
        data['features'][i]['properties'][k] = predicted[iso][k][0]
        
#%%
# Add to map
cn = gpd.GeoDataFrame.from_features(data["features"])
geometry = list(cn['geometry'])
cn = cn.drop (columns= ['geometry'])

cn = gpd.GeoDataFrame(cn, crs="EPSG:4326", geometry=geometry)
cn.to_file ('avian_canada.shp', index= False)
cn.to_file('avian_canada.json', encoding= 'utf-8', driver="GeoJSON")

#%%
# Prepare for GitHub
Regional = open('avian_canada.json', 'r')
content0 = Regional.read()
content0 = 'ca_regions = ' + content0

avian_western_canada = open('avian_western_canada.json', 'r')
content1 = avian_western_canada.read()
content1 = 'ca_western = ' + content1

avian_central_canada = open('avian_central_canada.json','r')
content2 = avian_central_canada.read()
content2 = 'ca_central = ' + content2

avian_atlantic_canada = open('avian_atlantic_canada.json','r')
content3 = avian_atlantic_canada.read()
content3 = 'ca_atlantic = ' + content3

avian_country_canada = open('avian_country_canada.json', 'r')
content4 = avian_country_canada.read()
content4 = 'ca_country = ' + content4

#-------------------------------------------------------

f0 = open("Regional.js", "w")
f0.write(content0)
f0.close()

f1 = open("avian_western_canada.js","w")
f1.write(content1)
f1.close()

f2 = open('avian_central_canada.js','w')
f2.write(content2)
f2.close()

f3 = open('avian_atlantic_canada.js', 'w')
f3.write(content3)
f3.close()

f4 = open('avian_country_canada.js','w')
f4.write(content4)
f4.close()

#%%
# Update Github
from github import Github
from datetime import date
import os

#%%
token = os.getenv("MY_SECRET_TOKEN")
g = Github(token)
repo = g.get_user().get_repo("acadic-portal.github.io")

file0 = repo.get_contents("/Avian/Regional.js")
file1 = repo.get_contents("/Avian/avian_western_canada.js")
file2 = repo.get_contents("/Avian/avian_central_canada.js")
file3 = repo.get_contents("/Avian/avian_atlantic_canada.js")
file4 = repo.get_contents("/Avian/avian_country_canada.js")

repo.update_file (file0.path, 'Last Updated: '+date.today().strftime("%d/%m/%Y"), content0, file0.sha)
repo.update_file (file1.path, 'Last Updated: '+date.today().strftime("%d/%m/%Y"), content1, file1.sha)
repo.update_file (file2.path, 'Last Updated: '+date.today().strftime("%d/%m/%Y"), content2, file2.sha)
repo.update_file (file3.path, 'Last Updated: '+date.today().strftime("%d/%m/%Y"), content3, file3.sha)
repo.update_file (file4.path, 'Last Updated: '+date.today().strftime("%d/%m/%Y"), content4, file4.sha)

#%%

dummy0 = repo.get_contents("/Avian/dummies/cases.csv")
cases0 = open('cases.csv', 'r')
dummy0_0 = cases0.read()
repo.update_file (dummy0.path, 'Last Updated: '+date.today().strftime("%d/%m/%Y"), dummy0_0, dummy0.sha)
if gt_success == True:
    dummy1 = repo.get_contents("/Avian/dummies/gt.csv")
    gt0 = open('gt.csv', 'r')
    dummy1_0 = gt0.read()
    repo.update_file (dummy1.path, 'Last Updated: '+date.today().strftime("%d/%m/%Y"), dummy1_0, dummy1.sha)
if gn_success == True:
    dummy2 = repo.get_contents("/Avian/dummies/gn1.csv") 
    gn0 = open('gn1.csv', 'r')
    dummy2_0 = gn0.read()
    repo.update_file (dummy2.path, 'Last Updated: '+date.today().strftime("%d/%m/%Y"), dummy2_0, dummy2.sha)
if r_success == True:
    dummy3 = repo.get_contents("/Avian/dummies/reddit2.csv") 
    r0 = open('reddit2.csv', 'r')
    dummy3_0 = r0.read()
    repo.update_file (dummy3.path, 'Last Updated: '+date.today().strftime("%d/%m/%Y"), dummy3_0, dummy3.sha)
if gdelt_success == True:
    dummy4 = repo.get_contents("/Avian/dummies/gdelt.csv") 
    gdelt0 = open('gdelt.csv', 'r')
    dummy4_0 = gdelt0.read()
    repo.update_file (dummy4.path, 'Last Updated: '+date.today().strftime("%d/%m/%Y"), dummy4_0, dummy4.sha)







