# -*- coding: utf-8 -*-
"""
Created on Wed June 26 16:40:49 2023
Temperature Whiplash：

1-02: The projected changes in the characteristics of temperature flips during 1961–2100 are examined using daily temperature outputs from the simulations of 17 climate models participating in the ScenarioMIP within the CMIP6 project. T
The temperature output over 1961–2014 utilizes the historical simulations, whereas the 2015–2100 period uses the future climate projection under various emission and socioeconomic scenarios represented by four Tier 1 experiments based on SSP-RCP scenarios, which are SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5.
This code is used to process CMIP6 datasets.

The outputs of daily near-surface air temperature from multi-model climate simulations are from the CMIP6 portal at https://esgf-node.llnl.gov/projects/cmip6/. 


@author: Sijia Wu (wusj7@mail2.sysu.edu.cn), Ming Luo (luom38@mail.sysu.edu.cn)

"""

#%% import libraries
import pandas as pd
import xarray as xr
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import warnings
import pymannkendall as mk
from scipy import stats
import gc
import glob
from tqdm import tqdm

warnings.filterwarnings("ignore")      
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'


## CMIP6 model需要补充2月29日的数据
def cmip6_datetime_process(ds, var = 'tas'):
    lat = ds.lat
    lon = ds.lon
    
    ds['time'] = ds.indexes['time'].to_datetimeindex() ###转换时间格式 
    ## 创建标准calendar的时间格式
    time = pd.date_range(start= str(ds['time.year'][0].values)+'-01-01', end = str(ds['time.year'][-1].values)+'-12-31', freq='1d')
    ## 创建一个新的dataset，用于存放补充好的数据
    ds_temp = xr.Dataset(data_vars = dict(var = (('time', 'lat', 'lon'), np.full([len(time), len(lat), len(lon)], np.nan))),
                         coords = dict(time = time, lat = lat.values, lon = lon.values))
    
    ## 除去2月29以外的数据，赋值成原始的数据
    ds_temp['var'].values[~((ds_temp['time.month']==2)&(ds_temp['time.day']==29))] = ds[var].values
    
    ## 找到所有年份的2月29日的位置，将其赋值为前一天与后一天的平均值
    idx = np.where((ds_temp['time.month']==2)&(ds_temp['time.day']==29))[0]
    ds_temp['var'].values[idx, :,:] = (ds_temp['var'].values[idx-1, :,:] + ds_temp['var'].values[idx+1, :,:])/2.0
    
    ds_temp=ds_temp.rename({'var':var})
    return ds_temp


def convert_lon(data):

    lon = (((data[get_lon_name(data)] + 180) % 360) - 180)   # 获取转变后的lon
    idx1 = np.where(lon<0)[0]
    idx2 = np.where(lon>=0)[0]
    data_convert = data.assign_coords(lon= xr.concat((lon[idx1], lon[idx2]), dim='lon'))  #重新分配lon
    data_convert.values = xr.concat((data_convert[:, idx1], data_convert[:,  idx2]), dim='lon')     
    
    return data_convert    
    del lon, idx1, idx2
    

def linear_trend(x, y):  
    mask = (~(np.isnan(x))) & (~(np.isnan(y)))
    
    # calculate the trend with 80% valid data
    if len(x[mask]) <= 0.2 * len(x):
        return np.nan, np.nan,np.nan, np.nan
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
        result = mk.yue_wang_modification_test(y[mask])
        # result = mk.hamed_rao_modification_test(y[mask])
        p_value1 = result.p
        # return slope, p_value1, intercept, std_err
        return slope, p_value1, intercept, std_err

def get_lon_name(ds_):
    for lon_name in ['lon', 'longitude']:
        if lon_name in ds_.coords:
            return lon_name
    raise RuntimeError("Couldn't find a longitude coordinate")
    
def get_lat_name(ds_):
    for lat_name in ['lat', 'latitude']:
        if lat_name in ds_.coords:
            return lat_name
    raise RuntimeError("Couldn't find a latitude coordinate")


def cool2warm_whiplash(flag, max_gap=5):
    """ Identify cold2hot temperature whiplash events
    
    Parameters
    ----------
    flag : temperature flag series (1: hot, -1: cold, 0: non-hot & non-cold)
    max_gap : the maximum gap between the start and end of the event
        
    
    Returns
    -------
    cold2hot_loc: two-column arrary with the first/second column marking the start/end of the whiplash event
    """
    
    ## length of the time series
    n = len(flag)
    
    ## index location of the identified cold2hot events
    cold2hot_loc = [] 
    
    ## search for cold2hot  events
    for i in range(0, n):
        if (flag[i] == -1):  # if cold extreme found
            is_cold = True
            is_hot = False

            # search for hot extreme in the following 'max_gap' days
            for j in range(i+1, min(i+max_gap+1, n)):
                if (flag[j] == 1):  # hot extreme
                    is_hot = True
                    break

            # if hot extreme found
            if ((is_cold == True) & (is_hot == True)):
                cold2hot_loc.append([i, j])
                i = j+1  # search the next event
        else:
            is_cold = False
            
    cold2hot_loc = np.asarray(cold2hot_loc) ## convert from list to array
    if (len(cold2hot_loc)<=0):
        return []

    ## remove the repeated event (i.e., having the same index for hot extreme)
    repeated_idx = np.where((cold2hot_loc[0:len(cold2hot_loc)-1, 1] - cold2hot_loc[1:len(cold2hot_loc), 1])==0)[0]
    cold2hot_loc = np.delete(cold2hot_loc, repeated_idx, 0)
    
    col_nan = np.full((len(cold2hot_loc), 1), np.nan)
    cold2hot_loc = np.c_[cold2hot_loc, col_nan, col_nan]
    
    for row in range (len(cold2hot_loc)):
        t_start = int(cold2hot_loc[row][0])
        t_end = int(cold2hot_loc[row][1])
        
        ## 获取cold-to-warm发生之前，出现-1的最后一天
        t_start_cold = np.array(np.where(flag[:t_start+1]>-1))[0][-1]+1
        
        ## 获取cold-to-warm之后，结束1的最后一天
        t_end_warm = t_end+np.array(np.where(flag[t_end:]<1))[0][0]-1
        
        cold2hot_loc[row][2] =  t_start_cold 
        cold2hot_loc[row][3] =  t_end_warm 
        
    return cold2hot_loc

def warm2cool_whiplash(flag, max_gap=5):
    """ Identify hot2cold temperature whiplash events
    
    Parameters
    ----------
    flag : temperature flag series (1: hot, -1: cold, 0: non-hot & non-cold)
    max_gap : the maximum gap between the start and end of the event
        
    
    Returns
    -------
    hot2cold_loc: two-column arrary with the first/second column marking the start/end of the whiplash event
    """
    
    ## length of the time series
    n = len(flag)
    
    ## index location of the identified hot2cold events
    hot2cold_loc = [] 
    
    ## search for hot2cold  events
    for i in range(0, n):
        if (flag[i] == 1):  # if hot extreme found
            is_cold = False
            is_hot = True

            # search for cold extreme in the following 'max_gap' days
            for j in range(i+1, min(i+max_gap+1, n)):
                if (flag[j] == -1):  # cold extreme
                    is_cold = True
                    break

            # if hot extreme found
            if ((is_hot == True) & (is_cold == True)):
                hot2cold_loc.append([i, j])
                i = j+1  # search the next event
        else:
            is_hot = False
            
    hot2cold_loc = np.asarray(hot2cold_loc) ## convert from list to array

    if (len(hot2cold_loc)<=0):
        return []

    ## remove the repeated event (i.e., having the same index for hot extreme)
    repeated_idx = np.where((hot2cold_loc[0:len(hot2cold_loc)-1, 1] - hot2cold_loc[1:len(hot2cold_loc), 1]) ==0)[0]
    hot2cold_loc = np.delete(hot2cold_loc, repeated_idx, 0)
    
    col_nan = np.full((len(hot2cold_loc), 1), np.nan)
    hot2cold_loc = np.c_[hot2cold_loc, col_nan, col_nan]
    
    for row in range (len(hot2cold_loc)):
        t_start = int(hot2cold_loc[row][0])
        t_end = int(hot2cold_loc[row][1])
        
        ## 获取warm-to-cold发生之前，出现1的最后一天
        t_start_warm = np.array(np.where(flag[:t_start+1]<1))[0][-1]+1

        ## 获取warm-to-cold之后，结束1的最后一天
        t_end_cold = t_end+np.array(np.where(flag[t_end:]>-1))[0][0]-1
        
        hot2cold_loc[row][2] =  t_start_warm 
        hot2cold_loc[row][3] =  t_end_cold
    
    return hot2cold_loc
        

def land_sea_mask(data, time = True, time_var = 'time', no_Antarctica = False):
    ## input xarray
    land_sea_mask = xr.open_dataset(r'D:\01PhD\00Data\04boundary\land_sea_mask.2.5x2.5.nc')   # ERA5 mask文件
    land_sea_mask = land_sea_mask.rename({'longitude':'lon', 'latitude':'lat'})
    if time == True:
        land_sea_mask_= np.tile(land_sea_mask['lsm'],(len(data[time_var]),1,1))
    else:
        land_sea_mask_ = land_sea_mask['lsm'][0,:,:].values
        
    land_sea_mask_[land_sea_mask_<=0] = np.nan
    land_sea_mask_[land_sea_mask_>0] = 1
    data = data* land_sea_mask_
    
    if no_Antarctica == True:
        # 将纬度在-60到90之间的南极洲区域设置为NaN
        mask = (data['lat'] < -60)# 创建南极洲的掩膜
        # 将南极洲区域设置为NaN
        data = data.where(~mask)
        
    return data



def detrend_with_rolling(data, window=15, order=3):
    
    lat_len = len(data['lat'])
    lon_len = len(data['lon'])       
    ## Assign memory to save the results
    data_detrended = data.copy(deep=True)
    data_detrended[:] = np.nan
    
    ## Perform the detrending process
    plusminus = window // 2
    for doy in tqdm(range(1, 367)):    
        
        # For a given 'day', obtain a window around that day.
        center_index = data.time.dt.dayofyear.isin(doy)
        window_doys = (np.arange(doy - plusminus - 1, doy + plusminus) % 366) + 1
        window_index = data.time.dt.dayofyear.isin(window_doys)
       
        # Get the data in the window and average it by time.year
        temp = data[window_index, :].groupby("time.year").mean()
    
        # Detrend by polynomial fitting
        x = temp["year"]
        for iy in range(lat_len):
            for ix in range(lon_len):
                # if mask[iy, ix] == False:
                #     continue]
                y = temp[:, iy, ix].values
                idx = np.isfinite(y)
                if sum(idx)< 0.8 * len(y):
                    continue 
                
                trend = np.polyval(np.polyfit(x[idx].values, y[idx], deg=order), x)
                idx_year = (data[center_index, iy, ix]["time.year"] - temp["year"].min()).values
                data_detrended[center_index, iy, ix] = data[center_index, iy, ix] - trend[idx_year]   

    return data_detrended



#%% [0] Input path and parameter
'''
Input data is daily data
Then we rolling mean this data
对原始的daily数据进行rolling mean是为了使得数据平滑，去除一些偶然出现的高值的影响。

'''
model_list = ['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','CanESM5',
              'CESM2-WACCM',"CESM2",'CNRM-CM6-1','FGOALS-g3','GFDL-ESM4',
              'CMCC-CM2-SR5','IPSL-CM6A-LR','INM-CM4-8','INM-CM5-0','MIROC6',
              'MPI-ESM1-2-HR','MRI-ESM2-0','NorESM2-LM']


rlzn_list = {'ACCESS-CM2': 'r1i1p1f1_gn','ACCESS-ESM1-5': 'r1i1p1f1_gn', 
             'AWI-CM-1-1-MR':'r1i1p1f1_gn','BCC-CSM2-MR':'r1i1p1f1_gn',
             'CanESM5':'r1i1p1f1_gn','CESM2':'r4i1p1f1_gn',
             'CESM2-WACCM':'r1i1p1f1_gn','CMCC-CM2-SR5':'r1i1p1f1_gn',
             'CNRM-CM6-1': 'r1i1p1f2_gr','EC-Earth3': 'r1i1p1f1_gr',
             'EC-Earth3-Veg':'r1i1p1f1_gr','FGOALS-g3': 'r1i1p1f1_gn',
             'GFDL-ESM4': 'r1i1p1f1_gr1','INM-CM4-8':'r1i1p1f1_gr1', 
             'INM-CM5-0':'r1i1p1f1_gr1','IPSL-CM6A-LR': 'r1i1p1f1_gr', 
              'MIROC6': 'r1i1p1f1_gn', 'MPI-ESM1-2-HR': 'r1i1p1f1_gn', 
              'MPI-ESM1-2-LR': 'r1i1p1f1_gn', 'MRI-ESM2-0': 'r1i1p1f1_gn',
               'NorESM2-LM': 'r1i1p1f1_gn'}
            
years_list = {'hist-aer':range(1961, 2021), 'hist-GHG':range(1961, 2021), 
              'hist-nat':range(1961, 2021), 'historical':range(1961, 2015),
              "historical_merge_ssp126":range(1961, 2101), 
              "historical_merge_ssp245":range(1961, 2101), 
              "historical_merge_ssp370":range(1961, 2101),
              "historical_merge_ssp585":range(1961, 2101),
              "ssp126":range(2015, 2101), "ssp245":range(2015, 2101), 
              "ssp370":range(2015, 2101),"ssp585":range(2015, 2101)} 

scenario_list =["historical_merge_ssp126","historical_merge_ssp245",
                "historical_merge_ssp370","historical_merge_ssp585"]

# scenario_list =["historical","ssp126", "ssp245", "ssp370","ssp585"]

region_list_Climate =['Global','Arid', 'Temperate','Tropical','Polar', 'Cold']

season_list = ["Annual", "DJF", "MAM", "JJA", "SON"]
region_list={1:'ANUZ',2:'CAN',3:'EFTA',4:'EU12-H',5:'EU12-M',6:'EU15',7:'JPKR',
             8:'MEA-H',9:'USA',10:'BRA',11:'CHN',12:'EEU-FSU',13:'EEU',14:'IDN',
             15:'LAM-M',16:'MEX',17:'OAS-M', 18:'RUS',19:'SAF',20:'SSA-M',21:'TUR',
             22:'CAS',23:'IND',24:'LAM-L',25:'NAF',26:'OAS-CPA',27:'OAS-L',
             28:'MEA-M',29:'PAK',30:'SSA-L', 31:'Global'}
region_name_list = [region_list[key] for key in region_list]
whiplash = ["W2C", "C2W"]
whiplash_names = {"W2C": "Hot-to-Cold",
                  "C2W": "Cold-to-Hot"}

whiplash_keys = ["Frequency", "Maximum Intensity", "Transition Duration"]; 
n_keys = len(whiplash_keys)

whiplash_keys_units = {"Frequency": "(events)", 
                       "Maximum Intensity": "(s.d.)",
                       "Transition Duration": "(days)"}

var_list = ['W2C Frequency','C2W Frequency',
            'W2C Maximum Intensity','C2W Maximum Intensity',
            'W2C Transition Duration','C2W Transition Duration']

ref_period = slice('1961-01-01', '1990-12-31')

cycle_window = 31
max_gap = 5
roll_window =max_gap
detrend_order = 3
std = 1
method = 'Method_2'
IPCC = 'AR6'
dataset_name = 'ERA5'

'''
method 2: 
    (1) 首先对输入的原始气温数据tas去趋势，获得Detrended tas
    (2) Detrended tas进行5d滑动平均
    (3）计算std, 获取Standardized tas
'''
print('method 2')
_var = 'tas'

detrended = 'detrended_rolling'
work_dir = "Z:/Public/5_Others/Temperature_whiplash_no_bias"+"//"+ method


#%%[1] Process raw data
#%%% [1-1] Detrended data
'''
Detrend raw data to obtain detrended tas

'''

path_data = "Z:/Public/1_Datasets/1_ClimateData/CMIP6/CMIP6_tas_day_mergetime_regrid/historical_merge_ssp/"

rlzn_list = {'ACCESS-CM2': 'r1i1p1f1_gn','ACCESS-ESM1-5': 'r1i1p1f1_gn', 
             'AWI-CM-1-1-MR':'r1i1p1f1_gn','BCC-CSM2-MR':'r1i1p1f1_gn',
             'CanESM5':'r1i1p1f1_gn','CESM2':'r4i1p1f1_gn',
             'CESM2-WACCM':'r1i1p1f1_gn','CMCC-CM2-SR5':'r1i1p1f1_gn',
             'CNRM-CM6-1': 'r1i1p1f2_gr','EC-Earth3': 'r1i1p1f1_gr',
             'EC-Earth3-Veg':'r1i1p1f1_gr','FGOALS-g3': 'r1i1p1f1_gn',
             'GFDL-ESM4': 'r1i1p1f1_gr1','INM-CM4-8':'r1i1p1f1_gr1', 
             'INM-CM5-0':'r1i1p1f1_gr1','IPSL-CM6A-LR': 'r1i1p1f1_gr', 
              'MIROC6': 'r1i1p1f1_gn', 'MPI-ESM1-2-HR': 'r1i1p1f1_gn', 
              'MPI-ESM1-2-LR': 'r1i1p1f1_gn', 'MRI-ESM2-0': 'r1i1p1f1_gn',
               'NorESM2-LM': 'r1i1p1f1_gn'}
var_raw = 'tas'

for model in model_list:
    for scenario in scenario_list:
        years=years_list[scenario]
        ## 使用glob.glob函数获取匹配通配符的文件
        _filename = glob.glob(path_data +'tas_day_'+ model +'_'+ scenario + '*.nc')[0]  
        print(_filename)
        ds = xr.open_dataset(_filename)
        ds = ds['tas'].to_dataset(name = var_raw)

        ds = ds.sel(time = slice(str(years[0])+'-01-01', str(years[-1])+'-12-31'))
        print(ds)
        
        ## 对于这部分的model需要补充2月29日的数据
        model_process=['BCC-CSM2-MR','CESM2','CESM2-WACCM','CMCC-CM2-SR5','CanESM5', 
                       'FGOALS-g3', 'GFDL-ESM4', 'INM-CM4-8','INM-CM5-0','NorESM2-LM']
        if model in model_process:
            ds = cmip6_datetime_process(ds, var=var_raw)
            
        ## 掩膜陆地
        ds[var_raw] = land_sea_mask(ds[var_raw], time = True, time_var = 'time', no_Antarctica = True)
        
        ## Detrend data
        
        print('method 2')
        ## Detrend standardized tas
        print('detrended_rolling')
        ## 对所有年每天前后15天的进行去趋势
        ds['Detrended '+_var] = detrend_with_rolling(ds[_var], window=cycle_window, order=detrend_order)
        
        
        ## Save detrended data
        # ds.to_netcdf(work_dir+'/01_Data/02_CMIP6_'+detrended+'//'+ model +'_'+ scenario + '_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order) +'_'+str(ds['time.year'][0].values) + '-' +str(ds['time.year'][-1].values) +'.nc')
        del ds
        gc.collect()
    # del daily_bias
    gc.collect()
