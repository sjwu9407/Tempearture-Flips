# -*- coding: utf-8 -*-
"""
Created on Wed June 26 16:40:49 2023
Temperature Whiplash：

6-01:Projected future changes of temperature flip
Draw Figure-3.


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
import copy
import regionmask
from matplotlib import ticker
warnings.filterwarnings("ignore")      
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'



# regional mean weighted by latitudes
def regional_mean(data):
    weights = np.sqrt(np.cos(np.deg2rad(data['lat'])).clip(0., 1.))
    weights.name = "weights"
    regional_mean = data.weighted(weights).mean(("lon", "lat"), skipna=True)
    return regional_mean

# regional mean weighted by latitudes
def regional_sum(data):
    weights = np.sqrt(np.cos(np.deg2rad(data['lat'])).clip(0., 1.))
    weights.name = "weights"
    regional_sum = data.weighted(weights).sum(("lon", "lat"), skipna=True)
    return regional_sum

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

def moving_std(data, window=15):
    threshold = data.groupby("time.dayofyear").std()
    threshold = xr.concat([threshold, threshold, threshold], dim='dayofyear')
    threshold = threshold.rolling(center=True,  dayofyear=window).mean()
    threshold = threshold[366:732, ...]
    return threshold


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


#%%[6] Calculate regionl and seasonal trend
years = range(1961,2101)
ds_cmip_mean=xr.open_dataset(work_dir +'/02_Result/02_CMIP6_result_'+detrended+'/03_multimodel/'+ 'regionmean_seasonal_'+detrended+'_'+_var+'_polyfit_order-'+
str(detrend_order)+'_'+'whiplash_event-'+str(max_gap)+'_'+str(years[0])+'-2100.nc')

rows = ['Slope', 'P_value']
# scenario_list =["historical", "hist-aer", "hist-GHG", "hist-nat", 
#                 "ssp126", "ssp245","ssp370","ssp585"]
scenario_list_ =["historical", "ssp126", "ssp245","ssp370","ssp585"]
season_list_ = ['DJF', 'MAM', 'JJA', 'SON']
i=0
for region in region_list_Climate:
    print(region)
    j=0
    for season in season_list_:
        print(season)
        k=0
        ds_cmip_mean_sel = ds_cmip_mean.sel(region=ds_cmip_mean['region']==region).sel(season=ds_cmip_mean['season']==season)
        ds_cmip_mean_sel=ds_cmip_mean_sel.squeeze()
        for scenario in scenario_list_:
            print(scenario)
            Trend = pd.DataFrame(data = np.full([len(rows), len(var_list)],np.nan), 
                                 index = rows, 
                                 columns = var_list)
            if scenario_list == ["historical_merge_ssp126","historical_merge_ssp245","historical_merge_ssp370","historical_merge_ssp585"]:
                if scenario == "historical":
                    ds_cmip_mean_sel_ = ds_cmip_mean_sel.sel(scenario=ds_cmip_mean_sel['scenario']=="historical_merge_ssp126").sel(year=slice(years[0], 2014)) 
                    ds_cmip_mean_sel_["scenario"] = ['historical']
                elif scenario in ["hist-aer", "hist-GHG", "hist-nat"]:
                    ds_cmip_mean_sel_ = ds_cmip_mean_sel.sel(scenario=ds_cmip_mean_sel['scenario']==scenario).sel(year=slice(years[0], 2014)) 
                else:
                    ds_cmip_mean_sel_ = ds_cmip_mean_sel.sel(scenario=ds_cmip_mean_sel['scenario']=="historical_merge_"+scenario).sel(year =slice(2015, 2100)) 
                    ds_cmip_mean_sel_["scenario"] = [scenario]
            elif scenario_list == ["historical","ssp126","ssp245","ssp370","ssp585"]:
                if scenario == "historical":
                    ds_cmip_mean_sel_ = ds_cmip_mean_sel.sel(scenario=ds_cmip_mean_sel['scenario']==scenario).sel(year=slice(years[0], 2014)) 
                elif scenario in ["hist-aer", "hist-GHG", "hist-nat"]:
                    ds_cmip_mean_sel_ = ds_cmip_mean_sel.sel(scenario=ds_cmip_mean_sel['scenario']==scenario).sel(year=slice(years[0], 2014)) 
                else:
                    ds_cmip_mean_sel_ = ds_cmip_mean_sel.sel(scenario=ds_cmip_mean_sel['scenario']==scenario).sel(year =slice(2015, 2100)) 
            
            ds_cmip_mean_sel_=ds_cmip_mean_sel_.squeeze()    
            for var_name in var_list:
                # print(var_name)
                variable_tmp = ds_cmip_mean_sel_[var_name].mean(dim="model")
        
                trend = linear_trend(variable_tmp.year,  variable_tmp)
                Trend[var_name] = [trend[0]*100, trend[1]] 
                Trend['scenario'] = scenario   
                del trend 
            Trend['scenario'] = scenario
            if k == 0:
                Trend_scenario = Trend
            else:
                Trend_scenario = pd.concat([Trend_scenario, Trend])
            k = k + 1   
            del Trend
            
        Trend_scenario['season']=season    
        if j == 0:
            Trend_season = Trend_scenario
        else:
            Trend_season = pd.concat([Trend_season, Trend_scenario])
        j = j + 1   
        del Trend_scenario
        
    Trend_season['region']=region       
    if i == 0:
        Trend_region = Trend_season
    else:
        Trend_region = pd.concat([Trend_region, Trend_season])
    i = i + 1   
    del Trend_season
del i, j, k
del scenario_list_


#%%[8] CMIP6 season plot
'''
分季节: long-term series and shading and boxplot of seasonal trend
'''
#%%%%[8-1] Processed Data
scenario_list_ =["historical", "ssp126", "ssp245","ssp370","ssp585"]
season_list_ = ['DJF', 'MAM', 'JJA', 'SON']
years=range(1961, 2101)
data =xr.Dataset(data_vars=dict(C2W_Frequency=(["scenario","season","year"], np.full([len(scenario_list_),len(season_list_),len(years)],np.nan)),
                                W2C_Frequency=(["scenario","season","year"], np.full([len(scenario_list_),len(season_list_),len(years)],np.nan)),
                                C2W_Transition_Intensity=(["scenario","season","year"], np.full([len(scenario_list_),len(season_list_),len(years)],np.nan)),
                                W2C_Transition_Intensity=(["scenario","season","year"], np.full([len(scenario_list_),len(season_list_),len(years)],np.nan)),
                                C2W_Maximum_Intensity=(["scenario","season","year"], np.full([len(scenario_list_),len(season_list_),len(years)],np.nan)),
                                W2C_Maximum_Intensity=(["scenario","season","year"], np.full([len(scenario_list_),len(season_list_),len(years)],np.nan)),
                                C2W_Transition_Duration=(["scenario","season","year"], np.full([len(scenario_list_),len(season_list_),len(years)],np.nan)),
                                W2C_Transition_Duration=(["scenario","season","year"], np.full([len(scenario_list_),len(season_list_),len(years)],np.nan)),
                                C2W_Maximum_Duration=(["scenario","season","year"], np.full([len(scenario_list_),len(season_list_),len(years)],np.nan)),
                                W2C_Maximum_Duration=(["scenario","season","year"], np.full([len(scenario_list_),len(season_list_),len(years)],np.nan)),
                                ),
                             coords=dict(scenario=scenario_list_,
                                         season=season_list_,
                                         year=(years)))

data = data.rename({"C2W_Frequency":"C2W Frequency", 
                    "W2C_Frequency":"W2C Frequency",
                    "C2W_Transition_Intensity":"C2W Transition Intensity", 
                    "W2C_Transition_Intensity":"W2C Transition Intensity",
                    "C2W_Maximum_Intensity":"C2W Maximum Intensity", 
                    "W2C_Maximum_Intensity":"W2C Maximum Intensity",
                    "C2W_Transition_Duration":"C2W Transition Duration", 
                    "W2C_Transition_Duration":"W2C Transition Duration",
                    "C2W_Maximum_Duration":"C2W Maximum Duration", 
                    "W2C_Maximum_Duration":"W2C Maximum Duration"})

ds_cmip_mean=xr.open_dataset(work_dir + '/02_Result/02_CMIP6_result_'+detrended+'/03_multimodel/'+ 'regionmean_seasonal_'+detrended+'_'+_var+'_polyfit_order-'+
str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) +'_'+str(years[0])+'-2100.nc')
ds_cmip_mean_ = ds_cmip_mean.sel(region=(ds_cmip_mean['region']=='Global')).squeeze().mean(dim='model')

for scenario in scenario_list_:
    for season in season_list_:
        ds_cmip_mean_season = ds_cmip_mean_.sel(season=(ds_cmip_mean_['season']==season)).squeeze()
        for var in var_list:
            if scenario == "historical":
                ds_cmip_mean_season_sel = ds_cmip_mean_season.sel(scenario=ds_cmip_mean_season['scenario']=="historical_merge_ssp126").sel(year=slice(years[0], 1990)).squeeze()
            elif scenario in ["hist-aer", "hist-GHG", "hist-nat"]:
                ds_cmip_mean_season_sel = ds_cmip_mean_season.sel(scenario=ds_cmip_mean_season['scenario']==scenario).sel(year=slice(years[0], 1990)).squeeze()
            else:
                ds_cmip_mean_season_sel = ds_cmip_mean_season.sel(scenario=ds_cmip_mean_season['scenario']=="historical_merge_"+scenario).sel(year =slice(2071, 2100)).squeeze()
        
            ds_cmip_mean_season_sel["scenario"] = [scenario]
            ds_cmip_mean_season_sel = ds_cmip_mean_season_sel.squeeze()
            data[var][data["scenario"]==scenario,
                      data["season"]==season,
                      ds_cmip_mean_season_sel.year-years[0]]= ds_cmip_mean_season_sel[var]
            del ds_cmip_mean_season_sel
        del ds_cmip_mean_season
data_annual = data.mean(dim='year')
data_sel_ = data_annual.to_dataframe().reset_index()


#%%%%[8-2] long-term series and shading
'''
多y轴图
按类型分两张图，Frequency，duration，intensity在一张图中

且每个指标对应右侧子图 

'''
color_list_blue = {'historical':'grey', 'hist-aer':'#3081B8',
                   'hist-GHG':'#F99A58', 'hist-nat':'#69BE63', 
                   'ssp126':'#81B4E3', 'ssp245':'#2671B2', 
                   'ssp370':'#1C5182', 'ssp585':'#153D61'}
 
color_list_red = {'historical':'grey', 'hist-aer':'#3081B8', 
                  'hist-GHG':'#F99A58', 'hist-nat':'#69BE63', 
                  'ssp126':'#FD9D9D', 'ssp245':'#FF5B5B', 
                  'ssp370':'#DA0404', 'ssp585':'#580017'}

# color_list_circle = {'W2C':['grey','#B4CCE4','#6999C9','#5088C0','#2770B4'],
#                     'C2W':['grey','#E0AEAE','#CE8080','#BB4D4D','#A3181B']}

label_list = {'historical':'Hist-ALL', 'hist-aer':'Hist-AER', 'hist-GHG':'Hist-GHG', 
              'hist-nat':'Hist-NAT', 'ssp126':'SSP126', 'ssp245':'SSP245', 
              'ssp370':'SSP370', 'ssp585':'SSP585'}

# ylim_list_circle = {'W2C Frequency':(1, 2.6), 'W2C Maximum Intensity':(2.4, 2.62), 'W2C Transition Duration':(3.8,4.35), 
#                     'C2W Frequency':(1, 2.6), 'C2W Maximum Intensity':(2.4, 2.62), 'C2W Transition Duration':(3.8,4.35)}

## Draw
ds_cmip_mean_sel = ds_cmip_mean.sel(region=ds_cmip_mean['region']=='Global').sel(season=ds_cmip_mean['season']=='Annual').squeeze()

fig,ax = plt.subplots(3,2, figsize=(11, 8.5), dpi=300)
# 调整子图之间的间隔
plt.subplots_adjust(wspace=0.7, hspace=0.15)
for i in range (0,2):
    for j in range(0, 3):
        print(i, j)
        var=whiplash[i]+' ' + whiplash_keys[j]
        for scenario in ['historical','ssp126','ssp245','ssp370','ssp585']:
            if scenario == "historical":
                ds_cmip_mean_sel_ = ds_cmip_mean_sel.sel(scenario=ds_cmip_mean_sel['scenario']=="historical_merge_ssp126").sel(year=slice(years[0], 2014)) 
                ds_cmip_mean_sel_["scenario"] = ['historical']
            elif scenario in ["hist-aer", "hist-GHG", "hist-nat"]:
                ds_cmip_mean_sel_ = ds_cmip_mean_sel.sel(scenario=ds_cmip_mean_sel['scenario']==scenario).sel(year=slice(years[0], 2014)) 
            else:
                ds_cmip_mean_sel_ = ds_cmip_mean_sel.sel(scenario=ds_cmip_mean_sel['scenario']=="historical_merge_"+scenario).sel(year =slice(2015, 2100)) 
                ds_cmip_mean_sel_["scenario"] = [scenario]
            ds_cmip_mean_sel_=ds_cmip_mean_sel_.squeeze() 
        
            ds_cmip_mean_sel_=ds_cmip_mean_sel_.squeeze() 
            variable_tmp = ds_cmip_mean_sel_[var].mean(dim="model")
            variable_tmp_std = ds_cmip_mean_sel_[var].std(dim="model")
    
            _Trend=Trend_region[(Trend_region['region'] == 'Global')&(Trend_region['season'] == season)]
            _slope = _Trend.iloc[0][var]
            _pval = _Trend.iloc[1][var]
            
            ## plot
            data = variable_tmp.to_dataframe()
            data_std = variable_tmp_std.to_dataframe()
            
            if i ==0:
                color_list = color_list_red
            else:
                color_list = color_list_red
                
            ax[j,i].fill_between(data.index,
                      (data[var]-1/2*data_std[var]),
                      (data[var]+1/2*data_std[var]),
                      alpha=0.2, facecolor=color_list[scenario])
            ax[j,i].plot(variable_tmp['year'], variable_tmp, 
                         color = color_list[scenario], alpha=1, linewidth = 1.2,
                         label= label_list[scenario] + '(' + '%+.3f' % (_slope) + ', p=' + '%.2f' % (_pval)+ ')')
            
            ax[j,i].set_xlabel('')
            ax[2,i].set_xlabel('Year',fontsize = 13)
            ax[j,0].set_ylabel(whiplash_keys[j] + ' '+whiplash_keys_units[whiplash_keys[j]],
                               fontsize = 13, fontweight='bold')
            ## 设置轴标签的字体大小
            ax[j,i].tick_params(axis = 'y', labelsize=12)
            ax[0,i].set_xticks([]) # 隐藏x轴刻度
            ax[1,i].set_xticks([]) # 隐藏x轴刻度
            ax[2,i].tick_params(axis = 'x',labelsize=12)
            ## y轴刻度保留1位小数点
            ax[j,i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
            
            ## 子图添加标题
            ax[0,i].set_title(["Hot-to-Cold flip", "Cold-to-Hot flip"][i], fontsize=16, 
                              weight='bold', loc='right')

            # ## 每张图对应的右侧子图
            a=ax[j,i].get_position()
            ax1 = fig.add_axes([a.xmax, a.ymin, 0.12, (a.ymax-a.ymin)])  ##left, bottom, width, height
            ## Dumbell 
            # color_list_circle = {'W2C':['silver','#B4CCE4','#6999C9','#5088C0','#2770B4'],
            #                     'C2W':['silver','#E0AEAE','#CE8080','#BB4D4D','#A3181B']}
            color_list_circle = {'W2C':['#9587B4','#E0AEAE','#CE8080','#BB4D4D','#A3181B'],
                                'C2W':['#9587B4','#E0AEAE','#CE8080','#BB4D4D','#A3181B']}
            sizes=[25, 45, 65, 85, 105]
            # sizes=[50, 80, 110, 130, 160]
            alphas=[0.7, 0.7, 0.7, 0.9, 1.0]
            # Plotting the points for each season
            for idx, season in enumerate(season_list_):
                data_sel_season = data_sel_[data_sel_['season']==season][var].reset_index()
                for k in range(5):
                    ax1.scatter(idx+0.3, data_sel_season[var][k], 
                                color=color_list_circle[whiplash[i]][k],
                                edgecolor = color_list_circle[whiplash[i]][k],
                                linewidth =0.5, alpha=alphas[k],
                                s=sizes[k], zorder=3) ##color=colors[idx],alpha=alphas[k],s=sizes[k],
                # Connect the points with a solid line
                ax1.plot([idx+0.3]*5, data_sel_season[var], linewidth =1.1,
                         c=color_list_circle[whiplash[i]][4],
                         linestyle = '--', alpha=0.5, zorder=2)
 
            ax1.set_ylabel('')
            ax1.set_xlabel('')
            ## y轴刻度保留1位小数点
            ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
            ax1.yaxis.tick_right()
            ax1.yaxis.set_label_position("right")
            ax1.tick_params(axis = 'y', labelsize=12)   
            ## Y轴在绘图右侧
            if j==2:
                ax1.set_xticks(range(len(season_list_)))
                ax1.set_xticklabels(season_list_)
                ax1.tick_params(axis = 'x',labelsize=12, rotation=30)
            else:
                ax1.set_xticks(range(len(season_list_)))
                ax1.set_xticklabels('')

figure_name = 'Figure-3-seasonal.' + 'CMIP6_ensmean-Weighted and seasonal mean trendmaxgap-'+  str(max_gap) + '_'+str(years[0])+'-2100.pdf'
# plt.savefig(work_dir+ '/03_Figure/02_CMIP6_figure_'+detrended+'//'+ 'whiplash_event-'+ str(max_gap) +'//'+\
#             figure_name, format="pdf", bbox_inches = 'tight')
plt.show() 


#%%[9] IPCC AR6 spatio-temporal trend
#%%%%[9-1] Calculate IPCC AR6 spatio-temporal trend
ds_cmip = xr.open_dataset(r'Z:\Public\5_Others\Temperature_whiplash_no_bias\Method_2\02_Result\02_CMIP6_result_detrended_rolling\03_multimodel\yearly_multimodel_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '_1961-2100.nc')
ds_cmip = ds_cmip.mean(dim='model')

## -- 利用AR6分区进行掩膜
if IPCC == 'AR6':
    IPCC_region = regionmask.defined_regions.ar6.land
    mask = IPCC_region.mask(ds_cmip, lon_name='lon', lat_name='lat')   ### 创建AR6的掩膜文件
elif IPCC == 'SREX':
    IPCC_region = regionmask.defined_regions.srex
    mask = IPCC_region.mask(ds_cmip, lon_name='lon', lat_name='lat')   ### 创建SREX的掩膜文件
    
scenario_list_ = ['historical', 'ssp126', 'ssp245', 'ssp370','ssp585']
ds_trend_AR6 =xr.Dataset(data_vars=dict(C2W_Frequency=(["scenario","IPCC_region"], np.full([len(scenario_list_),len(range(0,len(IPCC_region)))],np.nan)),
                                        W2C_Frequency=(["scenario","IPCC_region"], np.full([len(scenario_list_),len(range(0,len(IPCC_region)))],np.nan)),
                                        C2W_Maximum_Intensity=(["scenario","IPCC_region"], np.full([len(scenario_list_),len(range(0,len(IPCC_region)))],np.nan)),
                                        W2C_Maximum_Intensity=(["scenario","IPCC_region"], np.full([len(scenario_list_),len(range(0,len(IPCC_region)))],np.nan)),
                                        C2W_Transition_Duration=(["scenario","IPCC_region"], np.full([len(scenario_list_),len(range(0,len(IPCC_region)))],np.nan)),
                                        W2C_Transition_Duration=(["scenario","IPCC_region"], np.full([len(scenario_list_),len(range(0,len(IPCC_region)))],np.nan)),
                                        ),
                    coords=dict(scenario = scenario_list_,
                                IPCC_region=range(0,len(IPCC_region))))

ds_trend_AR6 = ds_trend_AR6.rename({"C2W_Frequency":"C2W Frequency", 
                                    "W2C_Frequency":"W2C Frequency",
                                    "C2W_Maximum_Intensity":"C2W Maximum Intensity", 
                                    "W2C_Maximum_Intensity":"W2C Maximum Intensity",
                                    "C2W_Transition_Duration":"C2W Transition Duration", 
                                    "W2C_Transition_Duration":"W2C Transition Duration",})
ds_pval_AR6 = copy.deepcopy(ds_trend_AR6)
for _scenario in scenario_list_:
    if _scenario == 'historical':
        ds_cmip_ = ds_cmip.sel(scenario=(ds_cmip["scenario"]=='historical_merge_ssp126')).squeeze()
        ds_cmip_ = ds_cmip_.sel(year=slice(1961, 2014))  ## historical
    elif _scenario in ['ssp126', 'ssp245', 'ssp370','ssp585']:
        ds_cmip_ = ds_cmip.sel(scenario=(ds_cmip["scenario"]=='historical_merge_'+_scenario)).squeeze()
        ds_cmip_ = ds_cmip_.sel(year=slice(2023, 2100))  ## future SSP
        
    for var in list(ds_cmip.keys()):
        print(var)
        for IPCC_region_ID in range(0,len(IPCC_region)): 
            if IPCC == 'AR6':
                region = IPCC_region[IPCC_region_ID].abbrev 
            elif IPCC == 'SREX':
                region = IPCC_region[IPCC_region_ID+1].abbrev 
            print(IPCC_region_ID, region)
            ## 提取每个IPCC_region的数据
            mask_lon_lat = np.where(mask == IPCC_region.map_keys(region))  ##将掩膜区域的格网经纬度位置挑出来
            ds_cmip_mask = ds_cmip_[var][:, np.unique(mask_lon_lat[0]), np.unique(mask_lon_lat[1])] ## 将属于该区域的值提取出来, ##经纬度用于构建meshgrid
            ## 区域平均    
            ds_cmip_mask_region = regional_mean(ds_cmip_mask)
            del ds_cmip_mask, mask_lon_lat
            
            trend = linear_trend(ds_cmip_mask_region.year,  ds_cmip_mask_region)
            
            ds_trend_AR6[var][ds_trend_AR6["scenario"]==_scenario,
                              ds_trend_AR6["IPCC_region"]==IPCC_region_ID]= 100 * trend[0]
            ds_pval_AR6[var][ds_pval_AR6["scenario"]==_scenario,
                             ds_pval_AR6["IPCC_region"]==IPCC_region_ID]= trend[1]

out_file_name = 'multimodel_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '_2023-2100.nc'
# ds_trend_AR6.to_netcdf(work_dir +'/02_Result/02_CMIP6_result_'+detrended+'/03_multimodel' +'//'+ IPCC+'_trend_' +out_file_name)  
# ds_pval_AR6.to_netcdf(work_dir +'/02_Result/02_CMIP6_result_'+detrended+'/03_multimodel' +'//'+ IPCC+'_pval_' +out_file_name)  
del ds_trend_AR6, ds_pval_AR6
gc.collect()


#%%%%[9-2] Draw IPCC spatio-temporal trend
grid = True
inplot = 'pieplot'
ds_cmip = xr.open_dataset(r'Z:\Public\5_Others\Temperature_whiplash_no_bias\Method_2\02_Result\02_CMIP6_result_detrended_rolling\03_multimodel\yearly_multimodel_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '_1961-2100.nc')
## -- 利用AR6分区进行掩膜
if IPCC == 'AR6':
    IPCC_region = regionmask.defined_regions.ar6.land
    mask = IPCC_region.mask(ds_cmip, lon_name='lon', lat_name='lat')   ### 创建AR6的掩膜文件
elif IPCC == 'SREX':
    IPCC_region = regionmask.defined_regions.srex
    mask = IPCC_region.mask(ds_cmip, lon_name='lon', lat_name='lat')   ### 创建SREX的掩膜文件
del ds_cmip
gc.collect()
clabel_list = {"Frequency": "(events/100yr)", 
              "Maximum Intensity": "(s.d./100yr)",
              "Transition Duration": "(days/100yr)"}

level_list = {"Frequency": 2, 
              "Maximum Intensity": 1,
              "Transition Duration": 0.8}

scenario_list_ = ['observation','ssp585']
color_list_blue = {'historical':'gray', 'hist-aer':'#3081B8',
                   'hist-GHG':'#F99A58', 'hist-nat':'#69BE63', 
                   'ssp126':'#81B4E3', 'ssp245':'#2671B2', 
                   'ssp370':'#1C5182', 'ssp585':'#153D61'}
 
color_list_red = {'historical':'gray', 'hist-aer':'#3081B8', 
                  'hist-GHG':'#F99A58', 'hist-nat':'#69BE63', 
                  'ssp126':'#FD9D9D', 'ssp245':'#FF5B5B', 
                  'ssp370':'#DA0404', 'ssp585':'#580017'}


for var in whiplash_keys:
    # Draw spatial distribution of trend
    fig = plt.figure(figsize=(11, 6), dpi=300)  
    for i in range(0, 2):
        for j in range(0,2):
            _type = whiplash[j]
            if scenario_list_[i] == 'observation':
                out_file_name = 'observation_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '.nc'
                ds_trend_AR6 = xr.open_dataset(work_dir +'/02_Result/01_Other_result_'+detrended+'/02_IPCC_'+IPCC+'_yearly' +'//'+ IPCC+'_trend_' +out_file_name)
                ds_pval_AR6= xr.open_dataset(work_dir +'/02_Result/01_Other_result_'+detrended+'/02_IPCC_'+IPCC+'_yearly' +'//'+ IPCC+'_trend_' +out_file_name)
                ds_trend_AR6_ = ds_trend_AR6.sel(dataset = (ds_trend_AR6['dataset']=='ERA5')).squeeze()
                ds_pval_AR6_ = ds_pval_AR6.sel(dataset = (ds_pval_AR6['dataset']=='ERA5')).squeeze()
            elif scenario_list_[i] == 'historical':
                out_file_name =  'multimodel_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '_2023-2100.nc'
                ds_trend_AR6 = xr.open_dataset(work_dir +'/02_Result/02_CMIP6_result_'+detrended+'/03_multimodel' +'//'+ IPCC+'_trend_' +out_file_name)
                ds_pval_AR6= xr.open_dataset(work_dir +'/02_Result/02_CMIP6_result_'+detrended+'/03_multimodel' +'//'+ IPCC+'_trend_' +out_file_name)
                ds_trend_AR6_ = ds_trend_AR6.sel(scenario = (ds_trend_AR6['scenario']=='historical')).squeeze()
                ds_pval_AR6_ = ds_pval_AR6.sel(scenario = (ds_pval_AR6['scenario']=='historical')).squeeze()
            if scenario_list_[i] == 'ssp585':
                out_file_name = 'multimodel_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '_2023-2100.nc'
                ds_trend_AR6 = xr.open_dataset(work_dir +'/02_Result/02_CMIP6_result_'+detrended+'/03_multimodel' +'//'+ IPCC+'_trend_' +out_file_name)
                ds_pval_AR6= xr.open_dataset(work_dir +'/02_Result/02_CMIP6_result_'+detrended+'/03_multimodel' +'//'+ IPCC+'_trend_' +out_file_name)
                ds_trend_AR6_ = ds_trend_AR6.sel(scenario = (ds_trend_AR6['scenario']=='ssp585')).squeeze()
                ds_pval_AR6_ = ds_pval_AR6.sel(scenario = (ds_pval_AR6['scenario']=='ssp585')).squeeze()
            
            AR6_trend = mask.to_dataset(name = _type+' ' +var)
            AR6_pval = mask.to_dataset(name = _type+' ' +var)
            
            # First, let's map the `temp` values to a dictionary for easy lookup
            ds_trend_AR6_values = ds_trend_AR6_[_type+' '+var].to_dict()['data']
            ds_pval_AR6_values = ds_pval_AR6_[_type+' '+var].to_dict()['data']
            
            # 计算大于0或者小于0的数的比例
            if var =='Transition Duration':
                positive_ratio_ = np.sum(np.array(ds_trend_AR6_values) < 0) / len(ds_trend_AR6_values)
            else:
                positive_ratio_ = np.sum(np.array(ds_trend_AR6_values) > 0) / len(ds_trend_AR6_values)
            # data_ratio[_type+' '+var][dataset_name] = positive_ratio_*100
            print(scenario_list_[i], _type+' '+var, positive_ratio_*100)
            # Now, iterate over each AR6_region value and replace the corresponding value in `ds`
            for region in ds_trend_AR6_['IPCC_region'].values:
                # The value to replace with
                ds_trend_AR6_value = ds_trend_AR6_values[region]
                ds_pval_AR6_value = ds_pval_AR6_values[region]
                if IPCC == 'AR6':
                    region_ = region
                elif IPCC == 'SREX':
                    region_ = region+1
                # Replace values in `ds` where the condition matches
                AR6_trend[_type+' '+var] = xr.where(AR6_trend[_type+' '+var]==region_, 
                                                    ds_trend_AR6_value, 
                                                    AR6_trend[_type+' '+var])
                AR6_pval[_type+' '+var] = xr.where(AR6_pval[_type+' '+var]==region_, 
                                                   ds_pval_AR6_value, 
                                                   AR6_pval[_type+' '+var])
            del ds_trend_AR6_, ds_pval_AR6_
            ## Draw
            ax = plt.subplot2grid((2, 2), (i, j), projection=ccrs.PlateCarree())
            plot_map_global(AR6_trend[_type+' ' +var], 
                            pvals = AR6_pval[_type+' ' +var],
                            ax = ax, pvals_style ='//',
                            levels= level_list[var]*np.arange(-1, 1.1, 0.1), 
                            ticks = level_list[var]*np.arange(-1, 1.1, 0.2), 
                            cmap=trunc_cmap, clabel = clabel_list[var], fraction=0.08, 
                            pad=0.1, alpha=0.1, linewidth=0.45,
                            title=scenario_list_[i], shift=True, cb = True,
                            grid=False, orientation="vertical")##horizontal

            if inplot == 'pdf':
                ## 每张大图内嵌的子图(PDF分布图)
                inset_ax = ax.inset_axes([0.035, 0.08, 0.25, 0.47]) 
                if grid:
                    if scenario_list_[i] == 'observation':
                        _filename = detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '.nc'
                        ds_trend=xr.open_dataset(work_dir +'/02_Result/01_Other_result_'+detrended+'/02_grid_seasonal' +'//'+'Grid_seasonal_trend_' +_filename)  
                        ds_trend = ds_trend.sel(season=(ds_trend['season']=="Annual")).squeeze()
                        color_list= {'ERA5':'red', 'Berkeley':'grey'}
                        for dataset_name in ['ERA5']:
                            trend = ds_trend[_type+' '+var][ds_trend["dataset"]==dataset_name,:,:].squeeze()
                            xs = calculate_pdf(trend)[0]
                            density_= calculate_pdf(trend)[1]
                            inset_ax.plot(xs,density_, label=dataset_name, color=color_list[dataset_name])
                            # # 去除NaN值
                            # trend = trend[~np.isnan(trend)]
                            # # 使用Seaborn的kdeplot函数绘制带阴影的PDF
                            # sns.kdeplot(trend, shade=True, label=dataset_name, color=color_list[dataset_name],ax = inset_ax)
                    elif scenario_list_[i] == 'historical':
                        _filename = 'multimodel_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '.nc'
                        ds_trend=xr.open_dataset(work_dir +'/02_Result/02_CMIP6_result_'+detrended+'/03_multimodel' +'//'+'Grid_trend_' +_filename)  
                        trend = ds_trend[_type+' '+var][ds_trend["scenario"]=='historical',:,:].values.flatten()           
                        # 去除NaN值
                        trend = trend[~np.isnan(trend)]
                        # 使用Seaborn的kdeplot函数绘制带阴影的PDF
                        sns.kdeplot(trend, shade=True, ax = inset_ax)
                    elif scenario_list_[i] in ['historical','ssp126', 'ssp245', 'ssp370','ssp585']:
                        _filename = 'multimodel_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '.nc'
                        ds_trend=xr.open_dataset(work_dir +'/02_Result/02_CMIP6_result_'+detrended+'/03_multimodel' +'//'+'Grid_trend_' +_filename)  
                        if _type == 'C2W':
                            color_list= color_list_red
                        else:
                            color_list= color_list_blue
                        for scenario in ['historical','ssp126', 'ssp245', 'ssp370','ssp585']:
                            # trend = ds_trend[_type+' '+var][ds_trend["scenario"]==scenario,:,:].values.flatten()
                            # # 去除NaN值
                            # trend = trend[~np.isnan(trend)]
                            # # 使用Seaborn的kdeplot函数绘制带阴影的PDF
                            # sns.kdeplot(trend, shade=False, label=str(scenario), color=color_list[scenario],ax = inset_ax)
                            trend = ds_trend[_type+' '+var][ds_trend["scenario"]==scenario,:,:].squeeze()
                            xs = calculate_pdf(trend)[0]
                            density_= calculate_pdf(trend)[1]
                            inset_ax.plot(xs,density_, label=scenario, color=color_list[scenario])
                    del _filename, ds_trend, trend
                else:
                    if scenario_list_[i] == 'observation':
                        color_list= {'ERA5':'red', 'Berkeley':'grey'}
                        for dataset_name in ['ERA5']:
                            trend = ds_trend_AR6[_type+' '+var][ds_trend_AR6["dataset"]==dataset_name,:].squeeze()
                            xs = calculate_pdf(trend)[0]
                            density_= calculate_pdf(trend)[1]
                            inset_ax.plot(xs,density_, label=dataset_name, color=color_list[dataset_name])
                    elif scenario_list_[i] in ['historical','ssp126', 'ssp245', 'ssp370','ssp585']:
                        if _type == 'C2W':
                            color_list= color_list_red
                        else:
                            color_list= color_list_blue
                        for scenario in ['ssp126', 'ssp245', 'ssp370','ssp585']:
                            trend = ds_trend_AR6[_type+' '+var][ds_trend_AR6["scenario"]==scenario,:].squeeze()
                            xs = calculate_pdf(trend)[0]
                            density_= calculate_pdf(trend)[1]
                            inset_ax.plot(xs,density_, label=scenario, color=color_list[scenario])
                    del  ds_trend_AR6, trend
                    
            elif inplot == 'barplot':
                # color_list_bar = {'W2C':'#81B4E3', 'C2W':'#FD9D9D'}
                color_list_bar = {'negative':'#81B4E3', 'positive':'#FD9D9D'}
                # 设置条形图的宽度
                width = 0.45
                
                # if scenario_list_[i] == 'observation':
                #     data_ratio = pd.DataFrame(index = ['ERA5','Berkeley'],columns=[_type+' '+var])
                #     for dataset_name in ['ERA5','Berkeley']:
                #         trend = ds_trend_AR6[_type+' '+var][ds_trend_AR6["dataset"]==dataset_name,:].squeeze()
                #         trend_AR6_values = trend.to_dict()['data']
                #         # 计算大于0或者小于0的数的比例
                #         if var =='Transition Duration':
                #             positive_ratio = np.sum(np.array(trend_AR6_values) < 0) / len(trend_AR6_values)
                #         else:
                #             positive_ratio = np.sum(np.array(trend_AR6_values) > 0) / len(trend_AR6_values)
                #         data_ratio[_type+' '+var][dataset_name] = positive_ratio*100
                        
                if scenario_list_[i] in ['historical','ssp126', 'ssp245', 'ssp370','ssp585']:
                    inset_ax = ax.inset_axes([0.035, 0.08, 0.25, 0.28]) 
                    scenario_list_sel = ['ssp126', 'ssp245', 'ssp370','ssp585']
                    positive_ratio = pd.DataFrame(index = ['ssp126','ssp245','ssp370','ssp585'],columns=[_type+' '+var])
                    negative_ratio = copy.deepcopy(positive_ratio)
                    for scenario in scenario_list_sel:
                        trend = ds_trend_AR6[_type+' '+var][ds_trend_AR6["scenario"]==scenario,:].squeeze()
                        trend_AR6_values = trend.to_dict()['data']
                        
                        positive_ratio[_type+' '+var][scenario] = (np.sum(np.array(trend_AR6_values) > 0) / len(trend_AR6_values))*100
                        negative_ratio[_type+' '+var][scenario] = (np.sum(np.array(trend_AR6_values) < 0) / len(trend_AR6_values))*100
                    
                    # inset_ax.bar(np.arange(len(scenario_list_sel))-width/2, 
                    #         positive_ratio[_type+' '+var], 
                    #         color = color_list_bar['positive'],width=width,
                    #         alpha=0.8) 
                    # inset_ax.bar(np.arange(len(scenario_list_sel))+width/2, 
                    #         negative_ratio[_type+' '+var], 
                    #         color = color_list_bar['negative'],width=width, alpha=0.8) 
                    
                    ## Stackplot
                    inset_ax.bar(np.arange(len(scenario_list_sel)), 
                            positive_ratio[_type+' '+var], 
                            color = color_list_bar['positive'],width=width,
                            alpha=0.8) 
                    inset_ax.bar(np.arange(len(scenario_list_sel)), 
                            negative_ratio[_type+' '+var], 
                            color = color_list_bar['negative'],width=width, 
                            bottom = positive_ratio[_type+' '+var],
                            alpha=0.8) 
                    
                    # sns.barplot(data=data_ratio, y=_type+' '+var, 
                    #             x=data_ratio.index, errcolor='none',
                    #             color  = color_list_bar[_type],
                    #             saturation = 0.5, 
                    #             width=0.45,ax = inset_ax) 
                    inset_ax.set_ylim([0,100])
                    inset_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
                    # 隐藏上边框和右边框
                    inset_ax.spines['top'].set_visible(False)
                    inset_ax.spines['right'].set_visible(False)
                    inset_ax.set_xticks(np.arange(len(scenario_list_sel)))
                    inset_ax.set_xticklabels(scenario_list_sel)
                    inset_ax.tick_params(axis='y', labelsize=8,pad=0.3, length=1)
                    inset_ax.tick_params(axis='x', labelsize=8,pad=0.5, length=1, rotation=20)
                    inset_ax.set_ylabel('')
                    inset_ax.set_xlabel('')
                    inset_ax.patch.set_alpha(0)  
            elif inplot == 'pieplot':
                labels = ['Significant Increase', 'Non-significant Increase', 
                          'Significant Decrease', 'Non-significant Decrease']
                colors = ['#FF5B5B','#FD9D9D','#2671B2','#81B4E3']
                explode = (0.1, 0, 0.1, 0)  # only "explode" the significant slices
                hatches = ['//','', '//', ''] # Empty for non-significant, diagonal lines for significant
                if scenario_list_[i] in ['historical','ssp126','ssp245','ssp370','ssp585']:
                    inset_ax = ax.inset_axes([0, 0.02, 0.2, 0.4]) 
                    scenario_list_sel = ['ssp126','ssp245','ssp370','ssp585']
                    for scenario in scenario_list_sel:
                        trend = ds_trend_AR6[_type+' '+var][ds_trend_AR6["scenario"]==scenario,:].squeeze().to_dataset(name = _type+' '+var)
                        pval = ds_pval_AR6[_type+' '+var][ds_pval_AR6["scenario"]==scenario,:].squeeze().to_dataset(name = _type+' '+var)
                        
                        significant_increase = ((pval[ _type+' '+var] <0.1) & (trend[ _type+' '+var] >= 0)).sum().item()
                        nonsignificant_increase = ((pval[ _type+' '+var] > 0.1) & (trend[ _type+' '+var] > 0)).sum().item()
                        significant_decrease = ((pval[ _type+' '+var] < 0.1) & (trend[ _type+' '+var] <= 0)).sum().item()
                        nonsignificant_decrease = ((pval[ _type+' '+var] > 0.1) & (trend[ _type+' '+var] <= 0)).sum().item()
                        
                        sizes = [significant_increase, nonsignificant_increase, 
                                 significant_decrease, nonsignificant_decrease]
                        print(sizes)
                        wedges, texts = inset_ax.pie(sizes, colors=colors, startangle=140)
                        # Adding the hatch patterns
                        for wedge, hatch in zip(wedges, hatches):
                            wedge.set_hatch(hatch)
                        # Equal aspect ratio ensures that pie is drawn as a circle
                        inset_ax.axis('equal')   
                        
    plt.tight_layout()
    # plt.savefig(work_dir+ '/03_Figure/02_CMIP6_figure_'+detrended+'//'+ 'whiplash_event-'+ str(max_gap) +'//'+ \
    #             'Figure-7-vertical.'+inplot+'-' + var+'-IPCC_'+IPCC+'_Observation_Historical_SSP585-Spatio-temperoal trend maxgap-'+  str(max_gap) + '_1961-2100.pdf',  
    #                 format="pdf", bbox_inches = 'tight', transparent=True)
    plt.show()
    del fig, i,j, ax, var
    gc.collect()


#%%%%[9-3-pieplot] Draw IPCC spatio-temporal trend
ds_cmip = xr.open_dataset(r'Z:\Public\5_Others\Temperature_whiplash_no_bias\Method_2\02_Result\02_CMIP6_result_detrended_rolling\03_multimodel\yearly_multimodel_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '_1961-2100.nc')
## -- 利用AR6分区进行掩膜
if IPCC == 'AR6':
    IPCC_region = regionmask.defined_regions.ar6.land
    mask = IPCC_region.mask(ds_cmip, lon_name='lon', lat_name='lat')   ### 创建AR6的掩膜文件
elif IPCC == 'SREX':
    IPCC_region = regionmask.defined_regions.srex
    mask = IPCC_region.mask(ds_cmip, lon_name='lon', lat_name='lat')   ### 创建SREX的掩膜文件
del ds_cmip
gc.collect()

out_file_name = 'multimodel_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '_2023-2100.nc'
ds_trend_AR6 = xr.open_dataset(work_dir +'/02_Result/02_CMIP6_result_'+detrended+'/03_multimodel' +'//'+ IPCC+'_trend_' +out_file_name)
ds_pval_AR6= xr.open_dataset(work_dir +'/02_Result/02_CMIP6_result_'+detrended+'/03_multimodel' +'//'+ IPCC+'_trend_' +out_file_name)

colors = ['#F7B799','#87BEDA']
fig,ax = plt.subplots(3,8, figsize=(16, 8), dpi=300)
# 调整子图之间的间隔
plt.subplots_adjust(wspace=0.1, hspace=0.1)
for i in range (0, 3):
    var = whiplash_keys[i]
    for j in range (0, 8):
        if j <4:
            _type = whiplash[0]
            scenario = ['ssp126', 'ssp245', 'ssp370','ssp585'][j]
        else:
            _type = whiplash[1]
            scenario = ['ssp126', 'ssp245', 'ssp370','ssp585'][j-4]
            
        ds_trend_AR6_ = ds_trend_AR6.sel(scenario = (ds_trend_AR6['scenario']==scenario)).squeeze()
        ds_pval_AR6_ = ds_pval_AR6.sel(scenario = (ds_pval_AR6['scenario']==scenario)).squeeze()
        
        trend = ds_trend_AR6[_type+' '+var][ds_trend_AR6["scenario"]==scenario,:].squeeze()
        trend_AR6_values = trend.to_dict()['data']
        
        positive_ratio = (np.sum(np.array(trend_AR6_values) > 0) / pd.Series(trend_AR6_values).dropna().shape[0])*100
        negative_ratio = (np.sum(np.array(trend_AR6_values) < 0) / pd.Series(trend_AR6_values).dropna().shape[0])*100
        
        sizes = [positive_ratio, negative_ratio]
        print(var, scenario, sizes)
        patches, texts, autotexts =ax[i,j].pie(sizes, 
                    autopct='%.1f%%',
                    colors=colors, 
                    startangle=140,
                    wedgeprops={'edgecolor':'w',#内外框颜色
                                'width': 0.45, 
                                'alpha':1,#透明度
                                })
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax[i,j].axis('equal')   
        
        ## 单独设置各组分的属性
        # patches[0].set_alpha(0.35)#A组分设置透明度

plt.legend(patches, ['Increase','Decrease'],#添加图例
          loc="lower center",ncol=2,frameon=False,
          fontsize=15)

plt.savefig(work_dir+ '/03_Figure/02_CMIP6_figure_'+detrended+'//'+ 'whiplash_event-'+ str(max_gap) +'//'+ \
            'Figure-7.pie_plot-IPCC_'+IPCC+'_different SSPs-Spatio-temperoal trend maxgap-'+  str(max_gap) + '_1961-2100.pdf',  
                format="pdf", bbox_inches = 'tight', transparent=True)
plt.show()
