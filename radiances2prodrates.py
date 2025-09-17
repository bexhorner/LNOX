#%%
# Import packages
%matplotlib inline
import numpy as np
import pandas as pd
import sys
import os
import glob
from netCDF4 import Dataset

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker

import cartopy
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

#%%
def get_lis_data(lis_file):
    ds = Dataset(lis_file, 'r')
    lons = ds['lon'][:]
    lats = ds['lat'][:]
    rad_hr = ds['rad_hr'][:]  # (lon, lat, hour)
    fl_hr = ds['fl_hr'][:]
    rad_24h = ds['rad_24h'][:]
    fl_24h = ds['fl_24h'][:]
    hours = np.arange(ds.dimensions['hours'].size)

    # Create DataFrame
    lon_grid, lat_grid, hour_grid = np.meshgrid(lons, lats, hours, indexing='ij')
    df_rad_hr = pd.DataFrame({
        'lon': lon_grid.flatten(),
        'lat': lat_grid.flatten(),
        'hour': hour_grid.flatten(),
        'rad_hr': rad_hr.flatten()
    })
    df_fl_hr = pd.DataFrame({
        'lon': lon_grid.flatten(),
        'lat': lat_grid.flatten(),
        'hour': hour_grid.flatten(),
        'fl_hr': fl_hr.flatten()
    })

    # 24h totals
    lon_grid_24, lat_grid_24 = np.meshgrid(lons, lats, indexing='ij')
    df_rad_24h = pd.DataFrame({
        'lon': lon_grid_24.flatten(),
        'lat': lat_grid_24.flatten(),
        'rad_24h': rad_24h.flatten()
    })
    df_fl_24h = pd.DataFrame({
        'lon': lon_grid_24.flatten(),
        'lat': lat_grid_24.flatten(),
        'fl_24h': fl_24h.flatten()
    })

    ds.close()

    df_rad_24h['rad_24h'].replace(0, np.nan, inplace=True)
    df_rad_hr['rad_hr'].replace(0, np.nan, inplace=True)

    rad_min = 64000
    rad_max = 1490000

    df_rad_hr['rad_hr'] = df_rad_hr['rad_hr'].clip(lower=rad_min, upper=rad_max)
    df_rad_24h['rad_24h'] = df_rad_24h['rad_24h'].clip(lower=rad_min, upper=rad_max)

    return df_fl_24h, df_rad_24h, df_fl_hr, df_rad_hr

#%%
def get_lis_data_all(df_trmm_24h, df_iss_24h, df_trmm_hr, df_iss_hr):
    # Merge 24h data
    df_24h = pd.merge(df_trmm_24h, df_iss_24h, on=['lon','lat'], suffixes=('_trmm','_iss'))

    # Compute mean across datasets
    df_24h['rad_24h'] = df_24h[['rad_24h_trmm','rad_24h_iss']].mean(axis=1)
    
    if 'fl_24h_trmm' in df_24h.columns and 'fl_24h_iss' in df_24h.columns:
        df_24h['fl_24h'] = df_24h[['fl_24h_trmm','fl_24h_iss']].mean(axis=1)
    else:
        df_24h['fl_24h'] = np.nan  # if flash counts are missing
    
    # Merge hourly data
    df_hr = pd.merge(df_trmm_hr, df_iss_hr, on=['lon','lat','hour'], suffixes=('_trmm','_iss'))
    df_hr['rad_hr'] = df_hr[['rad_hr_trmm','rad_hr_iss']].mean(axis=1)
    
    if 'fl_hr_trmm' in df_hr.columns and 'fl_hr_iss' in df_hr.columns:
        df_hr['fl_hr'] = df_hr[['fl_hr_trmm','fl_hr_iss']].mean(axis=1)
    else:
        df_hr['fl_hr'] = np.nan
    
    return df_24h[['lon','lat','rad_24h','fl_24h']], df_hr[['lon','lat','hour','rad_hr','fl_hr']]

#%%
def get_conversion_factor(y, avno, pnox, q_lis): #calculating beta factor from Wu et al. (2023)
    conversion_factor = (y/avno) * ( np.nanmean(q_lis*1e-6) / pnox )
    print('Conversion factor (/m2/sr/um) : ', conversion_factor)
    return conversion_factor

#%%
def get_diel_scaling(df_24h, df_hr):
    df = pd.merge(df_hr, df_24h, on=['lon','lat'], suffixes=('_hr','_24h'))
    df['sf'] = df['rad_hr'] / df.groupby(['lon','lat'])['rad_hr'].transform('sum')
    return df[['lon','lat','hour','sf']]

#%%
def interpolate_24h(df, value_col='rad_24h', method='nearest'):
    # Prepare points and values
    points = df[['lon','lat']].values
    values = df[value_col].values

    # Find NaNs
    mask_nan = np.isnan(values)
    if not mask_nan.any():
        return df.copy()

    # Interpolate only NaNs
    interpolated_values = griddata(
        points[~mask_nan], values[~mask_nan],
        points[mask_nan],
        method=method
    )

    df_interp = df.copy()
    df_interp.loc[mask_nan, value_col] = interpolated_values
    return df_interp

#%%
def interpolate_hr(df, value_col='rad_hr', method='nearest'):
    hours = df['hour'].unique()
    dfs = []

    for hr in hours:
        # Select only data for this hour
        df_hr = df[df['hour'] == hr].copy()

        # Call the existing 2D interpolate_24h function
        df_hr_interp = interpolate_24h(df_hr, value_col=value_col, method=method)

        # Add hour column back
        df_hr_interp['hour'] = hr

        dfs.append(df_hr_interp)

    # Combine all hours back into one DataFrame
    df_interp = pd.concat(dfs, ignore_index=True)
    return df_interp

#%%
def grid_2d(df, value_col='rad_24h', lon_res=0.625, lat_res=0.5):
    # Define bins
    lon_bins = np.arange(-180, 180+lon_res, lon_res)
    lat_bins = np.arange(-90, 90+lat_res, lat_res)
    
    df['lon_bin'] = pd.cut(df['lon'], bins=lon_bins, labels=lon_bins[:-1])
    df['lat_bin'] = pd.cut(df['lat'], bins=lat_bins, labels=lat_bins[:-1])
    
    # Average values per grid cell
    df_grid = df.groupby(['lon_bin','lat_bin'])[value_col].mean().reset_index()
    
    # Convert bins back to float
    df_grid['lon'] = df_grid['lon_bin'].astype(float)
    df_grid['lat'] = df_grid['lat_bin'].astype(float)
    df_grid = df_grid[[ 'lon','lat', value_col ]]
    
    return df_grid


def grid_3d(df, value_col='rad_hr', lon_res=0.625, lat_res=0.5):
    # Define lat/lon bins
    lon_bins = np.arange(-180, 180 + lon_res, lon_res)
    lat_bins = np.arange(-90, 90 + lat_res, lat_res)

    dfs = []
    for hr in sorted(df['hour'].unique()):
        df_hr = df[df['hour'] == hr].copy()
        
        # Bin lon/lat
        df_hr['lon_bin'] = pd.cut(df_hr['lon'], bins=lon_bins, labels=lon_bins[:-1])
        df_hr['lat_bin'] = pd.cut(df_hr['lat'], bins=lat_bins, labels=lat_bins[:-1])
        
        # Aggregate (mean) per grid cell
        df_grid = df_hr.groupby(['lon_bin','lat_bin'])[value_col].mean().reset_index()
        
        # Convert bin labels to float
        df_grid['lon'] = df_grid['lon_bin'].astype(float)
        df_grid['lat'] = df_grid['lat_bin'].astype(float)
        df_grid['hour'] = hr
        
        df_grid = df_grid[['lon','lat','hour',value_col]]
        dfs.append(df_grid)
    
    # Combine all hours
    df_all_hours = pd.concat(dfs, ignore_index=True)
    return df_all_hours

#%%
def get_lnox_prod(df, sfac, y=9e16, avno=6.022e23, q_col='rad_24h'):
    df_prod = df.copy()
    # Production rate formula
    df_prod['lnox'] = (y / (sfac * avno)) * (df_prod[q_col] * 1e-6)
    df_prod['lnox'] = df_prod['lnox'].fillna(260)
    print('Mean NOx prod (mol/fl): ', np.nanmean(df_prod['lnox']))
    
    return df_prod

#%%
def extend_and_fill_sf(df, lon_res=0.625, lat_res=0.5):
    # ---- Build full grid ----
    out_lon = np.arange(-180, 180, lon_res)
    out_lat = np.arange(-90, 90 + lat_res, lat_res)
    out_hour = np.arange(24)

    LON, LAT, HOUR = np.meshgrid(out_lon, out_lat, out_hour, indexing="ij")
    df_full = pd.DataFrame({
        "lon": LON.ravel(),
        "lat": LAT.ravel(),
        "hour": HOUR.ravel()
    })

    # ---- Merge with your sf df ----
    df_out = df_full.merge(df, on=["lon","lat","hour"], how="left")

    # ---- Fill hour by hour per (lon,lat) ----
    def normalize_group(group):
        valid_sum = group["sf"].sum(skipna=True)
        n_missing = group["sf"].isna().sum()

        if n_missing == 0:
            # Already complete → normalize to 1
            group["sf"] /= valid_sum if valid_sum > 0 else 1
        elif valid_sum > 0:
            # Some values exist → give remaining fraction equally to missing
            remaining = 1 - valid_sum
            group.loc[group["sf"].isna(), "sf"] = remaining / n_missing if n_missing > 0 else 0
        else:
            # All missing → uniform 1/24
            group["sf"] = 1/24

        return group

    df_out = df_out.groupby(["lon","lat"], group_keys=False).apply(normalize_group)

    return df_out.reset_index(drop=True)

#%%
def get_hr_prodrate(df_24h, df_sf, lon_res=0.625, lat_res=0.5):
    # ---- Ensure 24h prod has unique (lon,lat) ----
    df_24h = df_24h.copy()
    df_24h["lnox"] = df_24h["lnox"].fillna(260)   # fill missing with 260
    df_24h = df_24h.groupby(["lat","lon"], as_index=False).mean()

    # ---- Merge with scaling factors ----
    df_out = df_sf.merge(df_24h, on=["lon","lat"], how="left")

    # ---- Apply scaling factors ----
    df_out["lnox_hr"] = df_out["lnox"] * df_out["sf"]

    return df_out[["lon","lat","hour","lnox_hr","sf"]]
#%%

if __name__ == "__main__":

    #Define constants
    # Thermochemical NOx yield:
    tc_nox_yield = 9e16
    # Avogadro's Number (molec/mol):
    av_no = 6.022e23
    # Global mean NOx production per flash:
    gm_lpnox = 260

    lon_res = 0.625
    lat_res = 0.5

    lis_root = os.path.join('/home','ucfarh0','Lightning')
    filename_trmm = os.path.join(lis_root,'trmm_data.nc')
    filename_iss = os.path.join(lis_root,'iss_data.nc')

    lis_flashes_trmm, lis_radiances_trmm, lis_fl_hrly_trmm, lis_rad_hrly_trmm = get_lis_data(filename_trmm)
    lis_flashes_iss, lis_radiances_iss, lis_fl_hrly_iss, lis_rad_hrly_iss = get_lis_data(filename_iss)
    df_24h, df_hr = get_lis_data_all(lis_radiances_trmm, lis_radiances_iss, lis_rad_hrly_trmm, lis_rad_hrly_iss)
    
    df_sf_trmm = get_diel_scaling(lis_radiances_trmm, lis_rad_hrly_trmm)
    df_sf_iss = get_diel_scaling(lis_radiances_iss, lis_rad_hrly_iss)
    df_sf_all = get_diel_scaling(df_24h, df_hr)

    lis_radiances_trmm_int = interpolate_24h(lis_radiances_trmm)
    lis_radiances_iss_int = interpolate_24h(lis_radiances_iss)
    lis_radiances_all_int = interpolate_24h(df_24h)

    lis_rad_hrly_trmm_int = interpolate_hr(lis_rad_hrly_trmm)
    lis_rad_hrly_iss_int = interpolate_hr(lis_rad_hrly_iss)
    lis_rad_hrly_all_int = interpolate_hr(df_hr)

    lis_radiances_trmm_int = grid_2d(lis_radiances_trmm_int)
    lis_radiances_iss_int = grid_2d(lis_radiances_iss_int)
    lis_radiances_all_int = grid_2d(lis_radiances_all_int)

    lis_rad_hrly_trmm_int = grid_3d(lis_rad_hrly_trmm_int)
    lis_rad_hrly_iss_int = grid_3d(lis_rad_hrly_iss_int)
    lis_rad_hrly_all_int = grid_3d(lis_rad_hrly_all_int)

    scale_factor_trmm = get_conversion_factor(tc_nox_yield, av_no, gm_lpnox, lis_radiances_trmm_int['rad_24h'])
    scale_factor_iss = get_conversion_factor(tc_nox_yield, av_no, gm_lpnox, lis_radiances_iss_int['rad_24h'])
    scale_factor_all = get_conversion_factor(tc_nox_yield, av_no, gm_lpnox, lis_radiances_all_int['rad_24h'])

    prod_lnox_trmm = get_lnox_prod(lis_radiances_trmm_int, scale_factor_trmm)
    prod_lnox_iss = get_lnox_prod(lis_radiances_iss_int, scale_factor_iss)
    prod_lnox_all = get_lnox_prod(lis_radiances_all_int, scale_factor_all)

    prod_lnox_trmm = prod_lnox_trmm.sort_values(["lat", "lon"])
    prod_lnox_iss = prod_lnox_iss.sort_values(["lat", "lon"])
    prod_lnox_all = prod_lnox_all.sort_values(["lat", "lon"])

    #Get filled hourly scaling factors
    full_iss = extend_and_fill_sf(df_sf_iss)
    full_trmm = extend_and_fill_sf(df_sf_trmm)
    full_all = extend_and_fill_sf(df_sf_all)

    result_trmm = get_hr_prodrate(prod_lnox_trmm,full_trmm)
    result_iss = get_hr_prodrate(prod_lnox_iss,full_iss)
    result_all = get_hr_prodrate(prod_lnox_all,full_all)

    result_trmm = result_iss[result_trmm['lat'] < 90.0]
    result_iss = result_iss[result_iss['lat'] < 90.0]
    result_all = result_all[result_all['lat'] < 90.0]

    result_trmm = result_trmm.sort_values(["hour", "lat", "lon"])
    result_iss = result_iss.sort_values(["hour", "lat", "lon"])
    result_all = result_all.sort_values(["hour", "lat", "lon"])

    out_time = np.arange(0,24,1)
    zdim = len(out_time)

    out_lon = np.arange(-180, 180, 0.625) 
    out_lat = np.arange(-90, 90, 0.5)
    xdim = len(out_lon)
    ydim = len(out_lat)

    #Output 2D NetCDF
    ncout = Dataset('test_data.nc',mode='w',format="NETCDF4")
    #set time
    dt = np.datetime64('2020-11-29')
    end = np.timedelta64(0,'D')
    #Set dimensions
    ncout.createDimension('time', None)
    ncout.createDimension('lon', xdim)
    ncout.createDimension('lat', ydim)
    # #Set variables
    # #time
    t = ncout.createVariable('time', np.float32, ('time',))
    t.standard_name = 'time'
    t.units = 'days since 2020-1-1 00:00:00'
    t.calendar = 'gregorian'
    t.axis = 'T'
    t[:] = end
    # #longitide
    lon = ncout.createVariable('lon', np.float32, ('lon',))
    lon.standard_name = 'longitude'
    lon.long_name = 'Longitude'
    lon.units = 'degrees_east'
    lon.axis = 'X'
    lon[:] = out_lon
    # #latitude
    lat = ncout.createVariable('lat', np.float32, ('lat',))
    lat.standard_name = 'latitude'
    lat.long_name = 'Latitude'
    lat.units = 'degrees_north'
    lat.axis = 'Y'
    lat[:] = out_lat
    # #lnox production from TRMM
    lnox_prod_trmm = ncout.createVariable('PROD_LNOX_TRMM', np.float32, ('time' ,'lat', 'lon'), chunksizes=(1,ydim,xdim))
    lnox_prod_trmm.units = '1'
    lnox_prod_trmm.long_name = 'Production rate of lightning NOx from TRMM'
    lnox_prod_trmm[:] = (prod_lnox_trmm['lnox']*av_no).to_numpy().reshape(1,ydim,xdim)
    # #lnox production from ISS
    lnox_prod_iss = ncout.createVariable('PROD_LNOX_ISS', np.float32, ('time' ,'lat', 'lon'), chunksizes=(1,ydim,xdim))
    lnox_prod_iss.units = '1'
    lnox_prod_iss.long_name = 'Production rate of lightning NOx from ISS'
    lnox_prod_iss[:] = (prod_lnox_iss['lnox']*av_no).to_numpy().reshape(1,ydim,xdim)
    # #lnox production from both
    lnox_prod_all = ncout.createVariable('PROD_LNOX_ALL', np.float32, ('time' ,'lat', 'lon'), chunksizes=(1,ydim,xdim))
    lnox_prod_all.units = '1'
    lnox_prod_all.long_name = 'Production rate of lightning NOx from both TRMM and ISS'
    lnox_prod_all[:] = (prod_lnox_all['lnox']*av_no).to_numpy().reshape(1,ydim,xdim)
    ncout.close()

    #Output 3D NetCDF
    ncout = Dataset('test_data_diurnal.nc',mode='w',format="NETCDF4")
    out_time = np.arange(0,24,1)
    zdim = len(out_time)
    ncout.createDimension('time', zdim)
    ncout.createDimension('lon', xdim)
    ncout.createDimension('lat', ydim)
    #Set variables
    #time
    t = ncout.createVariable('time', np.float32, ('time',))
    t.standard_name = 'time'
    t.units = 'hours since 2020-1-1 00:00:00'
    t.calendar = 'gregorian'
    t.axis = 'T'
    t[:] = out_time
    #longitide
    lon = ncout.createVariable('lon', np.float32, ('lon',))
    lon.standard_name = 'longitude'
    lon.long_name = 'Longitude'
    lon.units = 'degrees_east'
    lon.axis = 'X'
    lon[:] = out_lon
    #latitude
    lat = ncout.createVariable('lat', np.float32, ('lat',))
    lat.standard_name = 'latitude'
    lat.long_name = 'Latitude'
    lat.units = 'degrees_north'
    lat.axis = 'Y'
    lat[:] = out_lat
    #lnox production from ISS
    lnox_prod_iss = ncout.createVariable('PROD_LNOX_ISS', np.float32, ('time', 'lat', 'lon'), chunksizes=(zdim,ydim,xdim))
    lnox_prod_iss.units = '1'
    lnox_prod_iss.long_name = 'Diurnal production rate of lightning NOx from ISS'
    lnox_prod_iss[:] = (np.array(result_iss['lnox_hr'])*av_no).reshape(zdim,ydim,xdim)
    #lnox production from TRMM
    lnox_prod_trmm = ncout.createVariable('PROD_LNOX_TRMM', np.float32, ('time', 'lat', 'lon'), chunksizes=(zdim,ydim,xdim))
    lnox_prod_trmm.units = '1'
    lnox_prod_trmm.long_name = 'Diurnal production rate of lightning NOx from TRMM'
    lnox_prod_trmm[:] = (np.array(result_trmm['lnox_hr'])*av_no).reshape(zdim,ydim,xdim)
    #lnox production from both
    lnox_prod_all = ncout.createVariable('PROD_LNOX_ALL', np.float32, ('time', 'lat', 'lon'), chunksizes=(zdim,ydim,xdim))
    lnox_prod_all.units = '1'
    lnox_prod_all.long_name = 'Diurnal production rate of lightning NOx from both TRMM and ISS'
    lnox_prod_all[:] = (np.array(result_all['lnox_hr'])*av_no).reshape(zdim,ydim,xdim)
    # #scaling factors
    lnox_sf_all = ncout.createVariable('SCALE_FACTOR_ALL', np.float32, ('time', 'lat', 'lon'), chunksizes=(zdim,ydim,xdim))
    lnox_sf_all.units = '1'
    lnox_sf_all.long_name = 'Diurnal scaling factors using TRMM and ISS'
    lnox_sf_all[:] = ((result_all['sf']).to_numpy()).reshape(zdim,ydim,xdim)
    lnox_sf_iss = ncout.createVariable('SCALE_FACTOR_ISS', np.float32, ('time', 'lat', 'lon'), chunksizes=(zdim,ydim,xdim))
    lnox_sf_iss.units = '1'
    lnox_sf_iss.long_name = 'Diurnal scaling factors using ISS'
    lnox_sf_iss[:] = ((result_iss['sf']).to_numpy()).reshape(zdim,ydim,xdim)
    lnox_sf_trmm = ncout.createVariable('SCALE_FACTOR_TRMM', np.float32, ('time', 'lat', 'lon'), chunksizes=(zdim,ydim,xdim))
    lnox_sf_trmm.units = '1'
    lnox_sf_trmm.long_name = 'Diurnal scaling factors using TRMM'
    lnox_sf_trmm[:] = ((result_trmm['sf']).to_numpy()).reshape(zdim,ydim,xdim)
    ncout.close()