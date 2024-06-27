# -*- coding: utf-8 -*-
"""
Created on Wed June 26 16:40:49 2023
Temperature Whiplash：

5-01:Observed historical changes of temperature flips.
This code is used to draw Figure-2


@author: Sijia Wu (wusj7@mail2.sysu.edu.cn), Ming Luo (luom38@mail.sysu.edu.cn)

"""

#%% import libraries
import xarray as xr
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings
import pymannkendall as mk
from scipy import stats
import gc
from tqdm import tqdm
import regionmask
import glob
import copy
from matplotlib import ticker
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
warnings.filterwarnings("ignore")      
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'

import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval,maxval,n)),
        )
    return new_cmap

# arr =np.linspace(0,50,100).reshape((10,10))

cmap = plt.get_cmap("RdBu_r")
trunc_cmap =truncate_colormap(cmap,0.2,0.8)


# regional mean weighted by latitudes
def regional_mean(data):
    weights = np.sqrt(np.cos(np.deg2rad(data['lat'])).clip(0., 1.))
    weights.name = "weights"
    regional_mean = data.weighted(weights).mean(("lon", "lat"))
    return regional_mean

# regional std weighted by latitudes
def regional_std(data):
    weights = np.sqrt(np.cos(np.deg2rad(data['lat'])).clip(0., 1.))
    weights.name = "weights"
    regional_std = data.weighted(weights).std(("lon", "lat"))
    return regional_std

    
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

from matplotlib.colors import LinearSegmentedColormap
def colormap():
    colors = [(1, 1, 1, 0), (0.5, 0.5, 0.5, 0), (1, 1, 1, 1)]
    return LinearSegmentedColormap.from_list('mycmap', colors)



# Plot contour map on the globe
def plot_map_global(data, pvals=None,
                    ax = plt.axes(projection=ccrs.PlateCarree()),
                    pvals_style ='maskout',
                    levels=np.arange(-1, 1.1, 0.1), 
                    ticks=np.arange(-1, 1.1, 0.3), 
                    cmap='RdYlBu_r', cb_position_x= 0.05,
                    clabel="", fraction=0.036, pad=0.1,alpha=0.05,linewidth=0.15,
                    title="", shift=False, grid=False, cb = True, extend = 'both',
                    xticks= True, yticks= True, title_fontsize = 12, IPCC_grid=False,
                    IPCC = 'AR6',
                    orientation="horizontal"):
    
    # shift coordinates from [0, 360] to [-180, 180]
    if shift:
        lon = (((data[get_lon_name(data)] + 180) % 360) - 180) 
        idx1 = np.where(lon<0)[0]
        idx2 = np.where(lon>=0)[0]
        data = data.assign_coords(lon= xr.concat((lon[idx1], lon[idx2]), dim='lon'))
        data.values = xr.concat((data[:, idx1], data[:, idx2]), dim='lon') 

    if False: # non-filled contour
        hc = data.plot.contour(ax=ax, transform=ccrs.PlateCarree(), 
                               x=get_lon_name(data), y=get_lat_name(data),
                               levels=levels, colors='k', linewidth=1.0)
    else: # filled contour 
        hc = data.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), 
                               x=get_lon_name(data), y=get_lat_name(data),
                               levels=levels, cmap=cmap, add_colorbar=False)
    if cb:
        # colorbar
        a=ax.get_position()
        pad=0.01
        height=0.02
        ax_f = fig.add_axes([cb_position_x +0.02, a.ymin - pad-0.02,  (a.xmax - a.xmin)*0.9 , height]) #长宽高
        cb = plt.colorbar(hc, ticks=levels, extend=extend, extendrect=False, extendfrac='auto',
                          orientation=orientation, fraction=fraction, pad=pad, cax=ax_f)
        # 设置colorbar的刻度标记为保留一位小数
        cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        cb.set_label(label=clabel, fontsize = 12,loc='center')
        cb.mappable.set_clim(min(levels), max(levels))
        cb.set_ticks(ticks)   ###设置colorbar的标注
        
        
    # plot signifiance (if have) over the map
    if (pvals is None):
        print('No significance hatched')
    else:
        if (pvals_style == 'maskout'):
            hp = pvals.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat',
                                      levels=[0, alpha], cmap=colormap(),
                                      add_colorbar=False, extend='both')     
        elif (pvals_style == 'dots'):
            hp = pvals.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), 
                                    x=get_lon_name(data), y=get_lat_name(data),hatches=[None, "..."],
                                      levels=[-1, alpha], colors='r', add_colorbar=False, extend='both', alpha=0)
                                        #hatches=[None, "xxx"],   
        elif (pvals_style == '//'):
            hp = pvals.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), 
                                    x=get_lon_name(data), y=get_lat_name(data),hatches=[None, "//"],
                                      levels=[-1, alpha], colors='r', add_colorbar=False, extend='both', alpha=0)
            # 设置hatch线条的粗细
            plt.rcParams['hatch.linewidth'] = 0.3  # 默认值是1.0
    # countries
    filename = r'D:\01PhD\00Data\04boundary\global\global.shp'
    countries = cfeat.ShapelyFeature(Reader(filename).geometries(), ccrs.PlateCarree(), facecolor=None)
    ax.add_feature(countries, facecolor='none', linewidth=linewidth, edgecolor='k')
    
    if IPCC_grid == True:
        if IPCC == 'AR6':
            filename = 'D:/01PhD/00Data/04boundary/IPCC_AR6/IPCC-AR6-Land.shp'   
            var = 'Acronym'
        elif IPCC == 'SREX':
            filename = 'D:/01PhD/00Data/04boundary/IPCC_SREX/IPCC-SREX-regions.shp'
            var = 'LAB'
        countries = cfeat.ShapelyFeature(Reader(filename).geometries(), ccrs.PlateCarree(), facecolor=None)
        ax.add_feature(countries, facecolor='none', linewidth=0.5, edgecolor='k')
        gdf = gpd.read_file(filename) 
        # 遍历GeoDataFrame来获取每个区域的中心和简称
        for idx, row in gdf.iterrows():
            # 获取几何形状的中心点
            centroid = row['geometry'].centroid
            # 假设简称存储在属性'short_name'中
            short_name = row[var]
            # 将简称添加到地图上
            ax.text(centroid.x, centroid.y, short_name, ha='center', va='center', 
                    fontsize=6, fontweight='bold',transform=ccrs.PlateCarree())
            
    # # set major and minor ticks
    # if xticks== 1:
    #     ax.set_xticks(np.arange(-180, 360+60, 60), crs=ccrs.PlateCarree())
    #     ax.xaxis.set_minor_locator(plt.MultipleLocator(30))
    #     ax.xaxis.set_major_formatter(LongitudeFormatter())
    # if yticks== 1:
    #     ax.set_yticks(np.arange(-90, 90+30, 30), crs=ccrs.PlateCarree())
    #     ax.yaxis.set_minor_locator(plt.MultipleLocator(30))
    #     ax.yaxis.set_major_formatter(LatitudeFormatter())
    # ax.tick_params(axis='both',which='major',labelsize=12)
    
    # show grid lines
    if grid:
        ax.grid(which='major', linestyle='--', linewidth=0.5, color=[0.25, 0.25, 0.25])     
    
    # set axis limits
    ax.set_ylim(-60, 90)
    ax.set_xlim(-180, 180)
    
    # set axis labels
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    
    ## 将坐标轴设置为不可见
    ax.axis('off')
    # set figure title
    ax.set_title(title, weight='bold', fontsize = title_fontsize)
    
    # return
    return hc, ax


#%% [0] Input path and parameter
'''
Input data is daily data
Then we rolling detrend this data
对原始的daily数据进行rolling mean是为了使得数据平滑，去除一些偶然出现的高值的影响。

'''
region_list = ['Global','Arid', 'Temperate','Tropical','Polar', 'Cold']
region_color = {'Arid': '#F2D599',
                'Temperate': '#D68582',
                'Tropical': '#B7ADE9',
                'Polar':'#79D9EF',
                'Cold':'#44A4A0'}
season_list = ["Annual", "DJF", "MAM", "JJA", "SON"]

dataset_name_list = ["ERA5","Berkeley","NCEP1"]  ##


whiplash = ["W2C", "C2W"]
whiplash_names = {"W2C": "Hot-to-Cold","C2W": "Cold-to-Hot"}

whiplash_keys = ["Frequency", "Maximum Intensity", "Transition Duration"]; 
n_keys = len(whiplash_keys)

relative_to_baseline=False
if relative_to_baseline==True:
    whiplash_keys_units = {"Frequency": "(%)", 
                           "Maximum Intensity": "(%)",
                           "Transition Duration": "(%)"}
    whiplash_keys_trend_units = {"Frequency": "(%/100yr)", 
                                 "Maximum Intensity": "(%/100yr)",
                                 "Transition Duration": "(%/100yr)"}
else:
    whiplash_keys_units = {"Frequency": "(events)", 
                           "Maximum Intensity": "(s.d.)",
                           "Transition Duration": "(days)"}
    whiplash_keys_trend_units = {"Frequency": "(events/100yr)", 
                                 "Maximum Intensity": "(s.d./100yr)",
                                 "Transition Duration": "(days/100yr)"}

dataset_startyear = {"ERA5": 1950,
                     "Berkeley": 1950,  
                     "NCEP1": 1948} 

dataset_endyear = {"ERA5": 2022,
                   "Berkeley": 2021,
                   "NCEP1": 2023} 

dataset_file_names = {"ERA5": "ERA5.global.2m_temperature.daily.2.5x2.5.1950_2022.nc",
                      "Berkeley": "Berkeley.global.tavg.19500101-20211231.2.5x2.5.nc",
                      "NCEP1": "NCEP1.global.2m_temperature.daily.2.5x2.5.1948_2023.nc"}

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


#%% [4] Regional mean
#%%% [4-2] Calculate regional mean
'''
获取某区域内的平均
先分配内存
"dataset", "region","season","year"

'''
years = range(1961, 2024)
ds_reanalysis_mean =xr.Dataset(data_vars=dict(C2W_Frequency=(["dataset","region","season","year"], np.full([len(dataset_name_list),len(region_list),len(season_list),len(years)],np.nan)),
                                              W2C_Frequency=(["dataset","region","season","year"], np.full([len(dataset_name_list),len(region_list),len(season_list),len(years)],np.nan)),
                                              C2W_Maximum_Intensity=(["dataset","region","season","year"], np.full([len(dataset_name_list),len(region_list),len(season_list),len(years)],np.nan)),
                                              W2C_Maximum_Intensity=(["dataset","region","season","year"], np.full([len(dataset_name_list),len(region_list),len(season_list),len(years)],np.nan)),
                                              C2W_Transition_Duration=(["dataset","region","season","year"], np.full([len(dataset_name_list),len(region_list),len(season_list),len(years)],np.nan)),
                                              W2C_Transition_Duration=(["dataset","region","season","year"], np.full([len(dataset_name_list),len(region_list),len(season_list),len(years)],np.nan)),
                                              ),
                               coords=dict(dataset=dataset_name_list,
                                           region=region_list,
                                           season=season_list,
                                           year=(years)))

ds_reanalysis_mean = ds_reanalysis_mean.rename({"C2W_Frequency":"C2W Frequency", 
                                                "W2C_Frequency":"W2C Frequency",
                                                "C2W_Maximum_Intensity":"C2W Maximum Intensity", 
                                                "W2C_Maximum_Intensity":"W2C Maximum Intensity",
                                                "C2W_Transition_Duration":"C2W Transition Duration", 
                                                "W2C_Transition_Duration":"W2C Transition Duration"})
climate_mask = xr.open_dataset(r'D:\01PhD\00Data\04boundary\Koppen\Koppen_climate_zone\Koppen_Climate_Zone.nc')

for dataset_name in dataset_name_list:
    _filename = glob.glob(work_dir + '/02_Result/01_Other_result_'+detrended+'/02_seasonal/whiplash_event-'+str(max_gap) +'//'+
                          'std-'+str(std)+'//'+ 'seasonal_' + dataset_name+'_'+detrended+'_'+_var+'_polyfit_order-'+
                          str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '*.nc')[0]
    print(_filename)
    ds_event = xr.open_dataset(_filename) 
    # ds_event = ds_event.sel(year=(ds_event['year']>=years[0]))
    ds_event = ds_event.sel(year=slice('1961', '2023'))
    
    for season in season_list:
        ds_event_sel = ds_event.sel(season=(ds_event['season']==season)).squeeze()
        for region in region_list:
            ## 按大洲的region进行掩膜
            if region == 'Global':
                print(season,region)
                ## 区域平均    
                ds_event_region = regional_mean(ds_event_sel)
            else:
                print(season,region)
                _mask = (climate_mask['Cli_Zone_mask'][climate_mask['Cli_Zone']==region,:,:]).squeeze()
                _mask = np.tile(_mask,(len(ds_event_sel['year']),1,1))
                ## 区域平均    
                ds_event_region = regional_mean(ds_event_sel * _mask)
                del _mask
            
            # ## 在计算DJF的时候会多出一年，所以将这个年份删除
            ds_event_region = ds_event_region.sel(year=(ds_event_region['year']<= (ds_event_sel.year.values.max()-1)))
            
            for var in list(ds_reanalysis_mean.keys()):
                ds_reanalysis_mean[var][ds_reanalysis_mean["dataset"]==dataset_name,
                                        ds_reanalysis_mean["region"]==region,
                                        ds_reanalysis_mean["season"]==season,
                                        ds_event_region.year-years[0]]= ds_event_region[var]
                
    del _filename, ds_event
    gc.collect()
    

#%%[7-8-2] Calculate Ensemble IPCC AR6 spatio-temporal trend
ds_ = xr.open_dataset(r'Z:/Public/5_Others/Temperature_whiplash_no_bias/Method_2/02_Result/01_Other_result_detrended_rolling/02_seasonal/whiplash_event-5/std-1/seasonal_ERA5_detrended_rolling_tas_polyfit_order-3_whiplash_event-5_1950-2022.nc') 
## -- 利用AR6分区进行掩膜
if IPCC == 'AR6':
    IPCC_region = regionmask.defined_regions.ar6.land
    mask = IPCC_region.mask(ds_, lon_name='lon', lat_name='lat')   ### 创建AR6的掩膜文件
elif IPCC == 'SREX':
    IPCC_region = regionmask.defined_regions.srex
    mask = IPCC_region.mask(ds_, lon_name='lon', lat_name='lat')   ### 创建SREX的掩膜文件
del ds_

ds_obser_trend_IPCC =xr.Dataset(data_vars=dict(C2W_Frequency=(["IPCC_region"], np.full([len(range(0,len(IPCC_region)))],np.nan)),
                                              W2C_Frequency=(["IPCC_region"], np.full([len(range(0,len(IPCC_region)))],np.nan)),
                                              C2W_Maximum_Intensity=(["IPCC_region"], np.full([len(range(0,len(IPCC_region)))],np.nan)),
                                              W2C_Maximum_Intensity=(["IPCC_region"], np.full([len(range(0,len(IPCC_region)))],np.nan)),
                                              C2W_Transition_Duration=(["IPCC_region"], np.full([len(range(0,len(IPCC_region)))],np.nan)),
                                              W2C_Transition_Duration=(["IPCC_region"], np.full([len(range(0,len(IPCC_region)))],np.nan)),
                                              ),
                               coords=dict(IPCC_region=range(0,len(IPCC_region))))
ds_obser_trend_IPCC = ds_obser_trend_IPCC.rename({"C2W_Frequency":"C2W Frequency", 
                                                "W2C_Frequency":"W2C Frequency",
                                                "C2W_Maximum_Intensity":"C2W Maximum Intensity", 
                                                "W2C_Maximum_Intensity":"W2C Maximum Intensity",
                                                "C2W_Transition_Duration":"C2W Transition Duration", 
                                                "W2C_Transition_Duration":"W2C Transition Duration",})
ds_obser_pval_IPCC = copy.deepcopy(ds_obser_trend_IPCC)

## Long-term Data
i = 0
for dataset_name in dataset_name_list:
    _filename = glob.glob(work_dir + '/02_Result/01_Other_result_'+detrended+'/02_seasonal/whiplash_event-'+str(max_gap) +'//'+
                          'std-'+str(std)+'//'+ 'seasonal_' + dataset_name+'_'+detrended+'_'+_var+'_polyfit_order-'+
                          str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '*.nc')[0]
    print(_filename)
    ds_event_ = xr.open_dataset(_filename) 
    ds_event_ = ds_event_.sel(year=slice('1961', '2023'))
    if i == 0:
        ds_event = ds_event_
    else:
        ds_event = xr.concat([ds_event,ds_event_],dim= 'dataset')
    i = i + 1
# 给新维度 'dataset' 添加坐标值，用于标识不同的数据集
ds_event = ds_event.assign_coords(dataset=dataset_name_list)
ds_event = ds_event.drop(['C2W Maximum Duration', 'W2C Maximum Duration', 
                          'C2W Transition Intensity', 'W2C Transition Intensity'])
ds_event = ds_event.mean(dim= 'dataset')
del i,ds_event_
ds_event = ds_event.sel(season=(ds_event['season']=='Annual')).squeeze()

for var in list(ds_obser_trend_IPCC.keys()):
    print(var)
    for IPCC_region_ID in range(0,len(IPCC_region)): 
        if IPCC == 'AR6':
            region = IPCC_region[IPCC_region_ID].abbrev 
        elif IPCC == 'SREX':
            region = IPCC_region[IPCC_region_ID+1].abbrev 
        print(region)
        ## 提取每个IPCC_region的数据
        mask_lon_lat = np.where(mask == IPCC_region.map_keys(region))  ##将掩膜区域的格网经纬度位置挑出来
        ds_obser_mask = ds_event[var][:, np.unique(mask_lon_lat[0]), np.unique(mask_lon_lat[1])] ## 将属于该区域的值提取出来, ##经纬度用于构建meshgrid
        ## 区域平均    
        ds_obser_mask_region = regional_mean(ds_obser_mask)
        del ds_obser_mask, mask_lon_lat
        
        trend = linear_trend(ds_obser_mask_region.year,  ds_obser_mask_region)
        
        ds_obser_trend_IPCC[var][ds_obser_trend_IPCC["IPCC_region"]==IPCC_region_ID]= 100 * trend[0]
        ds_obser_pval_IPCC[var][ds_obser_pval_IPCC["IPCC_region"]==IPCC_region_ID]= trend[1]

out_file_name = 'observation_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '.nc'
ds_obser_trend_IPCC.to_netcdf(work_dir +'/02_Result/01_Other_result_'+detrended+'/02_IPCC_'+IPCC+'_yearly/'+ IPCC+'_ensemble_trend_' +out_file_name)  
ds_obser_pval_IPCC.to_netcdf(work_dir +'/02_Result/01_Other_result_'+detrended+'/02_IPCC_'+IPCC+'_yearly/'+ IPCC+'_ensemble_pval_' +out_file_name)  
del ds_obser_trend_IPCC, ds_obser_pval_IPCC
gc.collect()


#%%[7-8-3] Draw Ensemble IPCC AR6 spatio-temporal trend
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

out_file_name = 'observation_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) + '.nc'
ds_obser_trend_IPCC = xr.open_dataset(work_dir +'/02_Result/01_Other_result_'+detrended+'/02_IPCC_'+IPCC+'_yearly/'+ IPCC+'_ensemble_trend_' +out_file_name)  
ds_obser_pval_IPCC = xr.open_dataset(work_dir +'/02_Result/01_Other_result_'+detrended+'/02_IPCC_'+IPCC+'_yearly/'+ IPCC+'_ensemble_pval_' +out_file_name)  

position_x_list = {"W2C": 0.0, 
                   "C2W": 0.5}

position_y_list = {"Frequency": 0.87, 
                   "Maximum Intensity": 0.53,
                   "Transition Duration": 0.19}

yticks = {"W2C Frequency": 1, 
          "C2W Frequency": 0,
          "W2C Maximum Intensity": 1,
          "C2W Maximum Intensity": 0,
          "W2C Transition Duration": 1,
          "C2W Transition Duration": 0}

xticks = {"W2C Frequency": 0, 
          "C2W Frequency": 0,
          "W2C Maximum Intensity": 0,
          "C2W Maximum Intensity": 0, 
          "W2C Transition Duration": 1,
          "C2W Transition Duration": 1}
level_list = {"Frequency": 2, 
              "Maximum Intensity": 1,
              "Transition Duration": 0.8}

## Long-term Data
region = 'Global'
ds_reanalysis_mean_sel = ds_reanalysis_mean.mean(dim='dataset')
ds_reanalysis_mean_sel = ds_reanalysis_mean_sel.sel(region=(ds_reanalysis_mean_sel['region']==region)).squeeze()

fig = plt.figure(figsize=(11, 8.5), dpi=300)   
for i in range(len(ds_obser_trend_IPCC.keys())):
    var = list(ds_obser_trend_IPCC.keys())[i]
    if var in ['C2W Frequency', 'W2C Frequency']:
        ax = plt.axes([position_x_list[var.split(" ")[0]],position_y_list[var.split(" ")[1]],0.4,0.4],projection=ccrs.PlateCarree())
        clabel=whiplash_keys_units[var.split(" ")[1]]
        level = level_list[var.split(" ")[1]]
    else:
        ax = plt.axes([position_x_list[var.split(" ")[0]],position_y_list[var.split(" ")[1]+' '+var.split(" ")[2]],0.4,0.4],projection=ccrs.PlateCarree())
        clabel=whiplash_keys_units[var.split(" ")[1]+' '+var.split(" ")[2]]
        level = level_list[var.split(" ")[1]+' '+var.split(" ")[2]]
    
    AR6_trend = mask.to_dataset(name =var)
    AR6_pval = mask.to_dataset(name = var)
    
    # First, let's map the `temp` values to a dictionary for easy lookup
    ds_trend_AR6_values = ds_obser_trend_IPCC[var].to_dict()['data']
    ds_pval_AR6_values = ds_obser_pval_IPCC[var].to_dict()['data']
    
    # 计算大于0或者小于0的数的比例
    if var in ['C2W Transition Duration', 'W2C Transition Duration']:
        positive_ratio_ = np.sum(np.array(ds_trend_AR6_values) < 0) / len(ds_trend_AR6_values)
    else:
        positive_ratio_ = np.sum(np.array(ds_trend_AR6_values) > 0) / len(ds_trend_AR6_values)
    print(var, positive_ratio_*100)
    # Now, iterate over each AR6_region value and replace the corresponding value in `ds`
    for region in ds_obser_trend_IPCC['IPCC_region'].values:
        # The value to replace with
        ds_trend_AR6_value = ds_trend_AR6_values[region]
        ds_pval_AR6_value = ds_pval_AR6_values[region]
        if IPCC == 'AR6':
            region_ = region
        elif IPCC == 'SREX':
            region_ = region+1
        # Replace values in `ds` where the condition matches
        AR6_trend[var] = xr.where(AR6_trend[var]==region_, 
                                            ds_trend_AR6_value, 
                                            AR6_trend[var])
        AR6_pval[var] = xr.where(AR6_pval[var]==region_, 
                                            ds_pval_AR6_value, 
                                            AR6_pval[var])
    plot_map_global(AR6_trend[var], pvals=AR6_pval[var],ax = ax,
                    levels=level*np.arange(-1, 1.1, 0.1), 
                    ticks = level*np.arange(-1, 1.1, 0.2),  
                    cmap=trunc_cmap, clabel=clabel, extend='both',
                    pvals_style ='//',IPCC_grid=True,
                    IPCC = 'AR6',fraction=0.08, pad=0.1,alpha=0.1,linewidth=0.35, 
                    cb_position_x=position_x_list[var.split(" ")[0]],
                    title=" ", shift=True, grid=False, title_fontsize = 13, 
                    xticks= xticks[var], yticks= yticks[var],
                    cb = True, orientation="horizontal")
    
    a=ax.get_position()
    
    ## 每张大图内嵌的子图
    ax2 = fig.add_axes([a.xmin-0.005, a.ymin+0.0, 0.1, 0.13])
    season_mean = ds_reanalysis_mean_sel
    _annua_mean = season_mean.sel(season=(season_mean['season']=='Annual'))[var].squeeze()
    _Trend = linear_trend(_annua_mean.year,  _annua_mean.values)
    trend_line = 1
    series_line = 1
    ax2.plot(_annua_mean.year, _annua_mean.values, 
                 color = '#2169CB', 
                 alpha=0.9, 
                 linewidth = series_line,
                 label='(' + '%+.3f' % (100*_Trend[0]) + ', p=' + '%.2f' % (_Trend[1])+ ')')
    sns.regplot(data =_annua_mean,
                x= _annua_mean.year, y = _annua_mean.values, 
                ci=90, scatter = None, 
                line_kws={'lw': trend_line, 
                          'color':'#2169CB',   ##'color':color_list_line[whiplash[i]],
                          'alpha': 1},
                ax = ax2)
    del _annua_mean,_Trend

    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax2.tick_params(axis='y', labelsize=10,pad=0.3, length=1)
    ax2.tick_params(axis='x', labelsize=9,pad=0.5, length=1, rotation=20)
    ax2.patch.set_alpha(0) # 设置背景透明度
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    ## 将部分坐标轴设置为不可见
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.legend(loc='upper left', fontsize=9, handlelength=0,
                   shadow = False, fancybox= True,framealpha=0.5, frameon=False)
    
plt.tight_layout()
# plt.savefig(work_dir+ '/03_Figure/01_Other_figure_'+detrended+'//'+ 'whiplash_event-'+ str(max_gap) +'//'+ \
#             'Figure-8.'+ var+'-Ensemble_IPCC_'+IPCC+'_Observation-Spatio-temperoal trend maxgap-'+  str(max_gap) + '_1961-2022.pdf',  
#                 format="pdf", bbox_inches = 'tight', transparent=True)

del fig, i, ax, var
gc.collect()