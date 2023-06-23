import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq,curve_fit
from numpy import vstack
import math
from pandas import DataFrame
import os
import glob
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import configparser
from keras import regularizers
from keras import optimizers
from keras.optimizers import RMSprop
from keras.optimizers import Adam

""" setup file """
# 取出相匹配的檔案
target_path=r'*.xls'
name_list=glob.glob((target_path))
f=DataFrame()
filename = "./info.ini"
# choose file with 150mm and 147mm or 149mm
name_box=['150mm','147mm','149mm']


""" Machine learning to fit curve """
def parameters(f):
  x_profile=['PosX_µm','X_Percent']
  y_profile=['PosY_µm','Y_Percent']

  # X/Y profile 
  x1=f[x_profile[0]] 
  y1=f[x_profile[1]] 
  x2=f[y_profile[0]] 
  y2=f[y_profile[1]] 
  find_value_1=math.e**(-2)
  find_value_2=0.5
  return find_value_1,find_value_2,x1,y1,x2,y2


# 超參數
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='sigmoid'))
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[0.0002,0.0004,0.0006,0.0008])), loss='mse', metrics=['accuracy'])
    return model

# tuner = RandomSearch_method(
#     build_model,
#     objective='mse',
#     max_trials=5,
#     executions_per_trial=3,
#     directory='filename',
#     project_name='text')


def ML(px,py):
    # 正則化參數
    lambda_reg=0.001
    # build model 
    model=Sequential()
    model.add(Dense(units=200,input_dim=1,activation='relu'))
    model.add(Dense(units=200,activation='relu'))
    model.add(Dense(units=200,activation='relu'))
    model.add(Dense(units=1,activation='linear'))
    optimizer=optimizers.Adam(learning_rate=0.0008)
    model.compile(optimizer=optimizer,loss='mse')
    model.summary()
    model.fit(px,py,epochs=200)
    # predicion
    y_pred=model.predict(px)

    return y_pred


""" Algorithm to Search_method proper value """
# normalized data
def normalize_y(input_y):
  n_y=(input_y-min(input_y))/(max(input_y)-min(input_y))
  return n_y

# fit function 
def Search_method(input_x,input_y,find_value):
  # build the pivot X to save left, right data
  pivot=input_x
  pivot_len=len(pivot)//2
  pivot_l, pivot_r = pivot[:pivot_len], pivot[(pivot_len+1):]
  # 找到 pivot_l,pivot_r 對應的 y 值
  pivot_l_y = [input_y[j] for j, x in enumerate(pivot_l)]
  pivot_r_y = [input_y[pivot_len+1+j] for j, x in enumerate(pivot_r)]
  # the target values
  find_Y=find_value
  # initialize min_l,min_r
  min_l,min_r=[],[]
  closed_x=[None,None]

  # left side
  for i in range(len(pivot_l)):
    if pivot_l_y[i] < find_Y or pivot_l_y[i] > find_Y:
      dist_l=abs(pivot_l_y[i]-find_Y)
      if not min_l or dist_l<min(min_l):
        min_l.append(dist_l)
        closed_x[0]=pivot_l[i]

  # right side    
  for i in range(len(pivot_r)):
      if pivot_r_y[i] < find_Y or pivot_r_y[i] > find_Y:
        dist_r=abs(pivot_r_y[i]-find_Y)

      if not min_r or dist_r<min(min_r):
        min_r.append(dist_r)
        closed_x[1]=pivot_r[pivot_len+1+i]

  return min(min_l),min(min_r),closed_x

# data result
def show_result(input_x,input_y,find_value_1,find_value_2):
    # position at 13.5%
    min_l1,min_r1,closed_x1=Search_method(input_x,input_y,find_value_1)
    value_x1=abs(float(closed_x1[0])-float(closed_x1[1]))
    # position at 50%
    min_l2,min_r2,closed_x2=Search_method(input_x,input_y,find_value_2)
    value_x2=abs(float(closed_x2[0])-float(closed_x2[1]))
    print("(13.5-percent) The min dist in left:%f, The min dist in right:%f , (50-percent)  The min dist in left:%f, The min dist in right:%f:" %(min_l1,min_r1,min_l2,min_r2))
    print("(13.5-percent) The proper X: %f,%f ; (50-percent)  The proper X: %f,%f" %(float(closed_x1[0]),float(closed_x1[1]),float(closed_x2[0]),float(closed_x2[1])))
    print("(13.5-percent) The spot diameter in X:%f, (50-percent) The spot diameter in X:%f" %(value_x1,value_x2))

    return  value_x1,value_x2

def run_ML(x1,y1,x2,y2):
  # normalized from [0,1]
  x1=normalize_y(x1) 
  x2=normalize_y(x2) 
  y1=normalize_y(y1) 
  y2=normalize_y(y2) 
  # prediction from ML
  y_pred_profile_x= ML(x1,y1)
  y_pred_profile_y= ML(x2,y2)
  return y1,y2,y_pred_profile_x,y_pred_profile_y

# write in ini file to save parameters
def write_preparation(x1,y_pred_profile_x,x2,y_pred_profile_y,find_value_1,find_value_2):
  print('150mm/147mm  X profile')
  value_x_profile_1,value_x_profile_2=show_result(x1,y_pred_profile_x,find_value_1,find_value_2)
  print('150mm/147mm  Y profile')
  value_y_profile_1,value_y_profile_2=show_result(x2,y_pred_profile_y,find_value_1,find_value_2)

  return value_x_profile_1,value_x_profile_2,value_y_profile_1,value_y_profile_2

""" config """
def cfg(config,input_name,value_x_profile_1,value_x_profile_2,value_y_profile_1,value_y_profile_2):
  config.add_section('Final X profile values'+input_name)
  config.set('Final X profile values'+input_name,'spot diameters 13.5%',str(value_x_profile_1))
  config.set('Final X profile values'+input_name,'spot diameters 50%',str(value_x_profile_2))
  config.add_section('Final Y profile values'+input_name)
  config.set('Final Y profile values'+input_name,'spot diameters 13.5%',str(value_y_profile_1))
  config.set('Final Y profile values'+input_name,'spot diameters 50%',str(value_y_profile_2))
  #  'a' 為複寫追加
  with open(filename, 'a') as configfile:
    config.write(configfile)


""" plot """
def imgplot(x_in,y_in,y_pred ,input_profile_name,Name):
 # plot the results
    fig=plt.figure(figsize=(7,5))
    plt.title('ML curve-fitting')
    plt.xlabel('displacement')
    plt.ylabel('normalized power')
    plt.scatter(x_in,y_in,label='data')
    plt.plot(x_in,y_pred,'r',label='predition')
    plt.legend()
    plt.savefig(f'./imgs/fitting-curve_{input_profile_name}_{Name}.png')

# read corresponding data
for idx in name_list:
    # profile name
    profile_name=['X-profile','Y-profile']
    f=pd.read_excel(idx,sheet_name='results')
    f_name = os.path.splitext(idx)[0]
    f_basename=os.path.basename(f_name)
    config = configparser.ConfigParser()
    config.add_section('fit xls files'+f_basename)
    config.set('fit xls files'+f_basename,'file list',str(idx))
    
    if any(name in f_name for name in name_box):
      # get input parameters
      find_value_1,find_value_2,x1,y1,x2,y2=parameters(f)
      # get the training result
      y1,y2,y_pred_profile_x,y_pred_profile_y=run_ML(x1,y1,x2,y2)
      # prepare to write in ini file
      value_x_profile_1,value_x_profile_2,value_y_profile_1,value_y_profile_2=write_preparation(x1,y_pred_profile_x,x2,y_pred_profile_y,find_value_1,find_value_2)
      # config function
      cfg(config,f_name,value_x_profile_1,value_x_profile_2,value_y_profile_1,value_y_profile_2)
      # plot the curve
      imgplot(x1,y1,y_pred_profile_x,profile_name[0], f_basename)
      imgplot(x2,y2,y_pred_profile_y,profile_name[1], f_basename)
