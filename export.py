import math
from pandas import DataFrame
import os
import glob
import pandas as pd
import numpy as np
import configparser
from pandas import ExcelWriter


""" Export to excel """
# Spot size calculation
# parameters 150mm
def export_dataframe(px_150_1,px_150_2, py_150_1,py_150_2,px_147_1,px_147_2, py_147_1,py_147_2):
    wavelength=1.31
    diff=3
    # calculation table para
    NA_x_FWHM=0.5*(abs(px_150_2/1000-px_147_2/1000)/diff)*1.699
    NA_y_FWHM=0.5*(abs(py_150_2/1000-py_147_2/1000)/diff)*1.699
    NA_x_e2=0.5*(abs(px_150_1/1000-px_147_1/1000)/diff)
    NA_y_e2=0.5*(abs(py_150_1/1000-py_147_1/1000)/diff)
    NA_avg_x=(NA_x_FWHM+NA_x_e2)/2
    NA_avg_y=(NA_y_FWHM+NA_y_e2)/2
    spot_size_x=(2*wavelength)/(math.pi*NA_avg_x)
    spot_size_y=(2*wavelength)/(math.pi*NA_avg_y)
    
    name_title=''
    multi_index = pd.MultiIndex.from_tuples([
        ('X', '150'), 
        ('Y', '147'),
        ('delta pos (mm)',diff),
        ('wavelength(um)',wavelength),
        ('spot diameter X',spot_size_x),
        ('spot diameter Y',spot_size_y)], 
        names=[name_title,'position(mm)'])
    data = {
        '1/e^2(x)': [ px_150_1 , px_147_1,'','','',''],
        'FWHM(x)': [px_150_2, px_147_2,'','','',''],
        '1/e^2(y)': [py_150_1, py_147_1,'','','',''],
        'FWHM(y)': [py_150_2, py_147_2,'','','',''],
        'NA(1/e^2)': [NA_x_e2,NA_y_e2,'','','',''],
        'NA(FWHM)': [NA_x_FWHM,NA_y_FWHM,'','','',''],
        'NA(avg)': [NA_avg_x,NA_avg_y,'','','','']
        }
    df = pd.DataFrame(data, index=multi_index)
    df.reset_index('position(mm)')
    return df

if __name__=='__main__':

    config = configparser.ConfigParser()
    config.read('C:/Users/Cloudlight Optics/Desktop/beam-pofile-loader/info.ini', encoding='utf-8')
    key_section_name_X=['Final X profile values']
    key_section_name_Y=['Final Y profile values']

    # 讀取含有特定關鍵字的數值
    sections = config.sections()

    # 定義用來儲存讀取結果的變數
    px_150_1 = None
    px_150_2 = None
    py_150_1 = None
    py_150_2 = None
    px_147_1 = None
    px_147_2 = None
    py_147_1 = None
    py_147_2 = None

    for section in sections:
        if any(name in section for name in key_section_name_X):
            if  '150mm' in section:
                px_150_1 = config[section].getfloat('spot diameters 13.5%')
                px_150_2 = config[section].getfloat('spot diameters 50%') 
            elif '147mm' in section:
                px_147_1 = config[section].getfloat('spot diameters 13.5%')
                px_147_2 = config[section].getfloat('spot diameters 50%')

        elif any(name in section for name in  key_section_name_Y):
                if '150mm' in section:
                    py_150_1 = config[section].getfloat('spot diameters 13.5%')
                    py_150_2 = config[section].getfloat('spot diameters 50%')
                    print(py_150_2)

                elif '147mm' in section:
                    py_147_1 = config[section].getfloat('spot diameters 13.5%')
                    py_147_2 = config[section].getfloat('spot diameters 50%')

    # 將DataFrame寫入Excel檔案中
    writer = pd.ExcelWriter('C:/Users/Cloudlight Optics/Desktop/dataframe_MFD.xlsx', engine='xlsxwriter')
    df=export_dataframe(px_150_1,px_150_2, py_150_1,py_150_2,px_147_1,px_147_2, py_147_1,py_147_2)
    df.to_excel(writer, index=True,sheet_name='MFD_result')
    writer.save()