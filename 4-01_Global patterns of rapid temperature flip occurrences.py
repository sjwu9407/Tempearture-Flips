# -*- coding: utf-8 -*-
"""
Created on Wed June 26 16:40:49 2023
Temperature Whiplash：

4-01:Global patterns of rapid temperature flip occurrences
This code is used to draw Figure-1

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
import glob

from matplotlib import ticker
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

#%%% [4-1] IPCC-AR6 plot
import geopandas as gpd
from matplotlib import pyplot as plt
IPCC= 'AR6'

# 加载Shapefile文件
if IPCC == 'AR6':
    filename = 'D:/01PhD/00Data/04boundary/IPCC_AR6/IPCC-AR6-Land.shp'   
    var = 'Acronym'
elif IPCC == 'SREX':
    filename = 'D:/01PhD/00Data/04boundary/IPCC_SREX/IPCC-SREX-regions.shp'
    var = 'LAB'
    
fig=plt.figure(figsize=(11, 7))#设置一个画板，将其返还给fig
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
countries = cfeat.ShapelyFeature(Reader(filename).geometries(), ccrs.PlateCarree(), facecolor=None)
ax.add_feature(countries, facecolor='none', linewidth=0.5, edgecolor='k')
gdf = gpd.read_file(filename) 
gdf = gdf[0:44]
# 遍历GeoDataFrame来获取每个区域的中心和简称
for idx, row in gdf.iterrows():
    # 获取几何形状的中心点
    centroid = row['geometry'].centroid
    # 假设简称存储在属性'short_name'中
    short_name = row[var]
    # 将简称添加到地图上
    ax.text(centroid.x, centroid.y, short_name, ha='center', va='center', 
            fontsize=12, fontweight='bold',transform=ccrs.PlateCarree())
    
# countries
ax.coastlines(edgecolor='white',linewidth=0.5)
ax.add_feature(cfeat.OCEAN) ##添加海洋背景
# ax.stock_img()#添加地球背景
# set major and minor ticks
ax.set_xticks(np.arange(-180, 360+60, 60), crs=ccrs.PlateCarree())
ax.xaxis.set_minor_locator(plt.MultipleLocator(30))
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.set_yticks(np.arange(-90, 90+30, 30), crs=ccrs.PlateCarree())
ax.yaxis.set_minor_locator(plt.MultipleLocator(30))
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.tick_params(axis='both',which='major',labelsize=14)

# set axis limits
plt.ylim(-60, 90)
plt.xlim(-180, 180)
plt.tight_layout()
plt.title("IPCC AR6 climate reference regions", fontsize = 20, fontweight='bold')
# plt.savefig(work_dir+ '/03_Figure/01_Other_figure_'+detrended+'//'+ 'whiplash_event-'+ str(max_gap) +'//'+'std-'+str(std)+'//'+\
#             'Figure-0.IPCC-AR6 WGI References Regions.pdf',format="pdf")
plt.show()


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


#%%% [5-1] Draw Global patterns of temperature flips
'''
Spatial distribution of frequency, intensity, speed

Input: yearly series
inplot 为内嵌的子图，参数为'displot', 'barplot','pieplot'

'''
inplot='seasonplot'
colors_red = ['#2671B2', '#2671B2','#2671B2','#2671B2','#2671B2']
colors_blue = ['#A31919','#A31919','#A31919','#A31919','#A31919']

ensemble = True
## Ensemble dataset
if ensemble:
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
    ds_yearly_mean = ds_event.sel(season=(ds_event['season']=='Annual')).mean(dim = 'year').squeeze()
    ds_yearly_mean = land_sea_mask(ds_yearly_mean, time = False, no_Antarctica = True) ## 掩膜掉海洋和南极洲
    ds_ensemble_mean = ds_yearly_mean.mean(dim= 'dataset')
    del i,ds_event_
    
if max_gap==3:
    level_list = {"Frequency": np.arange(0.1, 3.51, 0.1),
                  "Maximum Intensity": np.arange(3, 3.61, 0.02),
                  "Transition Duration": np.arange(2.4, 3.11, 0.02)}
    tick_list = {"Frequency": np.arange(0.1, 3.51, 0.5), 
                 "Maximum Intensity": np.arange(3, 3.61, 0.1),
                 "Transition Duration": np.arange(2.4, 3.11, 0.1)}
    bar_lim_list = {"Frequency": (0.4, 2.8), 
                    "Maximum Intensity": (3,3.5),
                    "Transition Duration": (2.5,3)}
    season_bar_lim_list = {"Frequency": (0.16, 0.55), 
                            "Maximum Intensity": (3.2, 3.4),
                            "Transition Duration": (2.7, 2.9)}

elif max_gap==5:
    if std ==0.5:
        level_list = {"Frequency": np.arange(0, 12, 0.774),
                      "Maximum Intensity": np.arange(2, 3, 0.065),
                      "Transition Duration": np.arange(2.9, 3.7, 0.0516)}
        tick_list = {"Frequency": np.arange(0, 12, 1.548), 
                     "Maximum Intensity": np.arange(2, 3, 0.13),
                     "Transition Duration": np.arange(2.9, 3.7, 0.103)}
        bar_lim_list = {"Frequency": (5, 9), 
                        "Maximum Intensity": (2, 3),
                        "Transition Duration": (2, 3.7)}
    elif std ==1:
        level_list = {"Frequency": np.arange(0.1, 3.51, 0.1), 
                      "Maximum Intensity": np.arange(3, 3.61, 0.02),
                      "Transition Duration": np.arange(3.5, 4.71, 0.04)}
        tick_list = {"Frequency": np.arange(0.1, 3.51, 0.5), 
                      "Maximum Intensity": np.arange(3, 3.61, 0.1),
                      "Transition Duration": np.arange(3.5, 4.71, 0.2)}
        bar_lim_list = {"Frequency": (0.8, 2.5), 
                        "Maximum Intensity": (2.7, 3.5),
                        "Transition Duration": (3.9, 4.4)}
        season_bar_lim_list = {"Frequency": (0.2, 0.55), 
                                "Maximum Intensity": (3.2, 3.36),
                                "Transition Duration": (4, 4.3)}
    elif std == 1.5:
        level_list = {"Frequency": np.arange(0, 0.6, 0.039), 
                      "Maximum Intensity": np.arange(3.8, 4.8, 0.065),
                      "Transition Duration": np.arange(4, 5.2, 0.078)}
        tick_list = {"Frequency": np.arange(0, 0.6, 0.078),
                     "Maximum Intensity": np.arange(3.8, 4.8, 0.13),
                     "Transition Duration": np.arange(4, 5.2, 0.156)}
        bar_lim_list = {"Frequency": (0, 0.5), 
                        "Maximum Intensity": (4, 4.25),
                        "Transition Duration": (4.2, 4.8)}
elif max_gap==7:
    level_list = {"Frequency": np.arange(0.1, 3.5, 0.1),
                  "Maximum Intensity": np.arange(2.9, 3.51,  0.02),
                  "Transition Duration": np.arange(4.7, 6.2, 0.05)}
    tick_list = {"Frequency": np.arange(0.1, 3.5, 0.5),
                 "Maximum Intensity": np.arange(2.9, 3.51,  0.1), 
                 "Transition Duration": np.arange(4.7, 6.2, 0.2)}
    bar_lim_list = {"Frequency": (0.4, 2.15), 
                    "Maximum Intensity": (3, 3.5),
                    "Transition Duration": (5, 5.8)}
    season_bar_lim_list = {"Frequency": (0.2, 0.47), 
                            "Maximum Intensity": (3.2, 3.36),
                            "Transition Duration": (5.3, 5.71)}
ylim_list = {"Frequency": 0.1, 
             "Maximum Intensity": 0.02,
             "Transition Duration": 0.005}

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

color_dataset = {"ERA5": '#694DE2', 
                 "Berkeley":'#769fcd',
                 "NCEP1":'#BDBDBD'}
alpha_dataset = {"ERA5": 0.2, 
                 "Berkeley":0.3,
                 "NCEP1":0.3}
fig = plt.figure(figsize=(11, 8.5), dpi=300)   
for i in range(len(ds_yearly_mean.keys())):
    var = list(ds_yearly_mean.keys())[i]
    if var in ['C2W Frequency', 'W2C Frequency']:
        ax = plt.axes([position_x_list[var.split(" ")[0]],position_y_list[var.split(" ")[1]],0.4,0.4],projection=ccrs.PlateCarree())
        levels= level_list[var.split(" ")[1]]
        ticks = tick_list[var.split(" ")[1]]
        clabel=whiplash_keys_units[var.split(" ")[1]]
        bar_lim = bar_lim_list[var.split(" ")[1]]
        season_bar_lim = season_bar_lim_list[var.split(" ")[1]]
    else:
        ax = plt.axes([position_x_list[var.split(" ")[0]],position_y_list[var.split(" ")[1]+' '+var.split(" ")[2]],0.4,0.4],projection=ccrs.PlateCarree())
        levels= level_list[var.split(" ")[1]+' '+var.split(" ")[2]]
        ticks = tick_list[var.split(" ")[1]+' '+var.split(" ")[2]]
        clabel=whiplash_keys_units[var.split(" ")[1]+' '+var.split(" ")[2]]
        bar_lim = bar_lim_list[var.split(" ")[1]+' '+var.split(" ")[2]]
        season_bar_lim = season_bar_lim_list[var.split(" ")[1]+' '+var.split(" ")[2]]
    plot_map_global(ds_ensemble_mean[var], pvals=None,
                    ax = ax,levels= levels, ticks = ticks, 
                    cmap=trunc_cmap, clabel=clabel, extend='both',
                    fraction=0.08, pad=0.1,alpha=0.05,linewidth=0.35, 
                    cb_position_x=position_x_list[var.split(" ")[0]],
                    title=" ", shift=True, grid=False, title_fontsize = 13, 
                    xticks= xticks[var], yticks= yticks[var],
                    cb = True, orientation="horizontal")
    a=ax.get_position()
    
    ## 每张大图对应的右侧子图
    ax1 = fig.add_axes([a.xmax, a.ymin, 0.07, (a.ymax-a.ymin)])  ##left, bottom, width, height
    for dataset in dataset_name_list:
        ds_yearly_mean_ = ds_yearly_mean.sel(dataset=(ds_yearly_mean['dataset']==dataset)).squeeze()
        _mean = ds_yearly_mean_[var].mean(dim = 'lon')
        _std = ds_yearly_mean_[var].std(dim = 'lon')
        x1 = _mean-_std
        x2 = _mean+_std
        ax1.plot(_mean, _mean.lat, linewidth=1.4, color = color_dataset[dataset])
        ax1.fill_betweenx(x1=x1, x2=x2, y=_mean.lat,color = color_dataset[dataset], alpha=alpha_dataset[dataset])
        del x1, x2
    ax1.tick_params(axis='x', labelsize=10,pad=0.5,length=1.5)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    ax1.tick_params(axis='y', labelsize=8,pad=0.3, length=1.5)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_yticklabels('') ## 不显示该子图的y轴标签
    # 延长y轴的刻度线
    ax1.tick_params(axis='y', which='both', length=3.5)  # 设置刻度线的长度为10
    ax1.margins(y=0)
    
    ## 每张大图内嵌的子图
    # 设置条形图的宽度
    width = 0.25
    ax2 = fig.add_axes([a.xmin+0.019, a.ymin+0.026, 0.092, 0.083])
    offsets = np.arange(3)*width-(3-1)* width / 2
    for season in season_list: 
        annu_mean=ds_reanalysis_mean.mean(dim = 'year').sel(region=(ds_reanalysis_mean['region']=='Global')).squeeze()
        annu_mean=annu_mean.sel(season=(annu_mean['season']!='Annual'))
        
    # 绘制每个数据集的条形图
    for i, dataset in enumerate(annu_mean['dataset'].values):
        plt.bar(np.arange(len(annu_mean.season)) + offsets[i], 
                annu_mean[var].sel(dataset=dataset), 
                width=width, color = color_dataset[dataset], alpha=0.8)
    ax2.set_xticks(np.arange(len(annu_mean.season)))
    ax2.set_xticklabels(annu_mean.season.values)
    ax2.set_ylim(season_bar_lim)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax2.tick_params(axis='y', labelsize=10,pad=0.3, length=1)
    ax2.tick_params(axis='x', labelsize=9,pad=0.5, length=1, rotation=20)
    ax2.patch.set_alpha(0) # 设置背景透明度
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    ## 将部分坐标轴设置为不可见
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
        
    # ax2.get_legend().remove() ## 删除图例
plt.tight_layout()
# plt.savefig(work_dir+ '/03_Figure/01_Other_figure_detrended_rolling'+'//'+ 'whiplash_event-'+ str(max_gap) +'//'+'std-'+str(std)+'//'+\
#             'Figure-1.'+inplot+'_' + dataset_name+ '-Climatology of metrics maxgap-'+  str(max_gap) + '_'+ \
#             str(dataset_startyear[dataset_name]) + '-' +str(dataset_endyear[dataset_name])+'.pdf',format="pdf", bbox_inches = 'tight')
plt.show()
del fig, i, ax, ds_yearly_mean, var
del level_list, tick_list
gc.collect()
