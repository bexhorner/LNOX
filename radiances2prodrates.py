#!/usr/bin/python

#%%
# Import packages
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
def get_lis_data(lis_file): #get LIS data from the file

    df = Dataset(lis_file, mode='r')

    # Read the variables into numpy arrays
    fl_24h = df.variables['fl_24h'][:]
    fl_hr = df.variables['fl_hr'][:]
    rad_24h = df.variables['rad_24h'][:]
    rad_hr = df.variables['rad_hr'][:]

    # Set 'rad_24h' to NaN where 'fl_24h' <= 0
    rad_24h = np.where(fl_24h <= 0, np.nan, rad_24h)
    
    # Set 'rad_hr' to NaN where 'fl_hr' <= 0
    rad_hr = np.where(fl_hr <= 0, np.nan, rad_hr)

    # Adjust radiances to threshold values that yield NOx production rates within
    # the min and max range of 30 to 700 mol / flash published by
    # Schumann and Huntrieser (2007):
    imin_24h, jmin_24h = np.where(rad_24h < 64000)
    rad_24h[imin_24h,jmin_24h] = 64000
    imax_24h, jmax_24h = np.where(rad_24h > 1490000)
    rad_24h[imax_24h,jmax_24h] = 1490000

    # Adjust radiances to threshold values that yield NOx production rates within
    # the min and max range of 30 to 700 mol / flash published by
    # Schumann and Huntrieser (2007):
    imin_hr, jmin_hr, kmin_hr = np.where(rad_hr < 64000)
    rad_hr[imin_hr,jmin_hr,kmin_hr] = 64000
    imax_hr, jmax_hr, kmax_hr = np.where(rad_hr > 1490000)
    rad_hr[imax_hr,jmax_hr,kmax_hr] = 1490000

    flashes = fl_24h
    radiances = rad_24h
    
    # Read latitude and longitude
    lat = df.variables['lat'][:]
    lon = df.variables['lon'][:]
    
    # Close the dataset
    df.close()
    
    return flashes, radiances, fl_hr, rad_hr, lat, lon

#%%
def get_lis_data_all(lis_file_trmm,lis_file_iss): #Get average of TRMM and ISS data
    flashes_trmm, radiances_trmm, fl_hrly_trmm, rad_hrly_trmm, lat_trmm, lon_trmm = get_lis_data(lis_file_trmm)
    flashes_iss, radiances_iss, fl_hrly_iss, rad_hrly_iss, lat_iss, lon_iss = get_lis_data(lis_file_iss)

    flashes_all = np.nanmean(np.array([flashes_trmm, flashes_iss]), axis=0)
    radiances_all = np.nanmean(np.array([radiances_trmm, radiances_iss]), axis=0)
    fl_hrly_all = np.nanmean(np.array([fl_hrly_trmm, fl_hrly_iss]), axis=0)
    rad_hrly_all = np.nanmean(np.array([rad_hrly_trmm, rad_hrly_iss]), axis=0)

    return flashes_all, radiances_all, fl_hrly_all, rad_hrly_all

#%%
def get_conversion_factor(y, avno, pnox, fl_lis, q_lis): #calculating beta factor from Wu et al. (2023)
    conversion_factor = (y/avno) * ( np.nanmean(q_lis*1e-6) / pnox )
    print('Conversion factor (/m2/sr/um) : ', conversion_factor)
    return conversion_factor

#%%
def get_lnox_prod(y, sfac, avno, q_lis, fl_lis): #Calculate production rates
    plnox = ( y / ( sfac * avno ) ) * ( q_lis * 1e-6 )
    print('Mean NOx prod (mol/fl): ', np.nanmean(plnox))
    return plnox

#%%
def get_diel_scaling(rads_24h, hrly_rads): #Get hourly scaling factor from radiances

    # Define output data:
    diel_scale_factors = np.zeros_like(hrly_rads)

    # Loop over data to get scale factors:
    for i in np.arange(24):
        diel_scale_factors[:,:,i] = hrly_rads[:,:,i] / np.sum(hrly_rads, axis=2)

    return diel_scale_factors

#%%
def interpolate_24h(array): #Interpolate 24h production rate data using nearest neighbour algorithm

    # Find the indices of non-NaN values
    non_nan_indices = np.argwhere(~np.isnan(array))  # Indices of non-NaN values
    non_nan_values = array[~np.isnan(array)]  # Non-NaN values
    
    # Find the indices of NaN values
    nan_indices = np.argwhere(np.isnan(array))
    
    # Create a grid for the input array
    grid_x, grid_y = np.mgrid[0:array.shape[0], 0:array.shape[1]]
    
    # Interpolate the NaN values using the nearest neighbour method
    interpolated_values = griddata(
        points=non_nan_indices,
        values=non_nan_values, 
        xi=(grid_x, grid_y), 
        method='nearest'
    )
    
    # Fill the NaN locations in the filled_array with interpolated values
    array[nan_indices[:, 0], nan_indices[:, 1]] = interpolated_values[nan_indices[:, 0], nan_indices[:, 1]]
    
    return array

#%%
def interpolate_hr(array): #Interpolate hr production rate data using nearest neighbour algorithm

    # Define the grid based on array dimensions
    x = np.arange(array.shape[0])
    y = np.arange(array.shape[1])
    z = np.arange(array.shape[2])

    # Create the interpolator only for non-NaN values
    non_nan_indices = np.argwhere(~np.isnan(array))
    non_nan_values = array[~np.isnan(array)]
    interpolator = RegularGridInterpolator((x, y, z), array, bounds_error=False, fill_value=np.nan)
    
    # Find NaN indices and interpolate only these points
    nan_indices = np.argwhere(np.isnan(array))
    filled_values = interpolator(nan_indices)

    # Replace NaNs with interpolated values
    array[tuple(nan_indices.T)] = filled_values

    return array

#%%
def gridding(x,y,z,lon_res,lat_res): #Gridding LNOx data onto the GEOS-Chem input grid

    out_lon = np.arange(-180, 180, lon_res) 
    out_lat = np.arange(-90, 90+lat_res, lat_res)

    xdim = len(out_lon)
    ydim = len(out_lat)

    data = None
    data = [[[] for n in range(ydim)] for m in range(xdim)]

    for i,j,k in zip(x,y,z):
        p = np.argmin(abs(out_lon - i))
        q = np.argmin(abs(out_lat - j))
        data[p][q].append(k)

    n = []
    for x in data:
        for y in x:
            n.append(np.nanmean(y))

    lon = []
    lat = []
    for a in out_lon:
        for b in out_lat:
            lon.append(a)
            lat.append(b)

    return n,lon,lat,xdim,ydim,out_lon,out_lat

#%%
def get_24h_prodrate(lis_lon,lis_lat,prod_lnox): #Get pandas dataframe of 24h production rates

    lon = []
    lat = []
    for i in lis_lon.flatten():
        for j in lis_lat.flatten():
            lon.append(i)
            lat.append(j)

    df = pd.DataFrame()
    df['lon'] = lon
    df['lat'] = lat
    df['lnox'] = prod_lnox.flatten()

    return df

#%%
def get_filled_24h(lon_new,lat_new,lnox_new): #Fill 24h values outside LIS latitudes with 265 mol/flash
    df_new = pd.DataFrame()
    df_new['lon'] = lon_new
    df_new['lat'] = lat_new
    df_new['lnox'] = lnox_new
    df_new['lnox'] = df_new['lnox'].fillna(260)
    df_new = df_new.groupby(['lat','lon']).mean()
    return df_new

#%%
def get_hr_sf(lis_lon, lis_lat, diel_scale_factors, lon_res=0.625, lat_res=0.5):
    """
    Create a complete lon-lat-time grid with lnox scale factors.

    Parameters
    ----------
    lis_lon : array-like
        Array of longitudes where diel_scale_factors are defined.
    lis_lat : array-like
        Array of latitudes where diel_scale_factors are defined.
    diel_scale_factors : array-like
        Scale factors matching shape (len(lis_lon) * len(lis_lat) * 24).
    lon_res : float
        Longitude resolution of full grid.
    lat_res : float
        Latitude resolution of full grid.

    Returns
    -------
    full : pandas.DataFrame
        DataFrame with full lon-lat-time coverage and filled lnox values.
    """
    # ---- Build the sf DataFrame from inputs ----
    out_time = np.arange(24)
    LON, LAT, TIME = np.meshgrid(lis_lon, lis_lat, out_time, indexing="ij")

    df_sf = pd.DataFrame({
        "lon": LON.ravel(),
        "lat": LAT.ravel(),
        "time": TIME.ravel(),
        "lnox": diel_scale_factors.flatten()
    })

    # ---- Build the full global grid ----
    out_lon = np.arange(-180, 180, lon_res)
    out_lat = np.arange(-90, 90 + lat_res, lat_res)
    out_time = np.arange(24)

    LON_full, LAT_full, TIME_full = np.meshgrid(out_lon, out_lat, out_time, indexing="ij")

    df_full = pd.DataFrame({
        "lon": LON_full.ravel(),
        "lat": LAT_full.ravel(),
        "time": TIME_full.ravel(),
        "lnox": np.nan
    })

    # ---- Merge and fill ----
    full = pd.concat([df_full, df_sf])
    full = full.groupby(["time", "lat", "lon"]).mean().reset_index()

    # Fill missing lnox values: if no values exist for (lat, lon), set to 1
    full["grouped_sum"] = full.groupby(["lat", "lon"])["lnox"].transform("sum")
    mask = full["lnox"].isna() & (full["grouped_sum"] == 0)
    full.loc[mask, "lnox"] = 1/24

    return full.drop(columns=["grouped_sum"])

#%%
def get_hr_prodrate(lon,lat,lnox,diel_scale_factors,lon_sf,lat_sf,time_sf): #Get hr production rates
    
    #Get 24h production rate data
    df_new = pd.DataFrame()
    df_new['lon'] = lon
    df_new['lat'] = lat
    df_new['lnox'] = lnox
    df_new['lnox'] = df_new['lnox'].fillna(260)
    df_new = df_new.groupby(['lat','lon']).mean() 

    #Get hr scaling factors
    df_sf_new = pd.DataFrame()
    df_sf_new['lon'] = lon_sf
    df_sf_new['lat'] = lat_sf
    df_sf_new['time'] = time_sf
    df_sf_new['sf'] = diel_scale_factors

    #Apply scaling factors to 24h data to get hr production rate data
    out_lon = np.arange(-180, 180, lon_res) 
    out_lat = np.arange(-90, 90+lat_res, lat_res)
    out_time = np.arange(0,24,1)
    reshaped_sf = np.array(df_sf_new['sf']).reshape(len(out_time),len(out_lat),len(out_lon))
    reshaped = np.array(df_new['lnox']).reshape(len(out_lat),len(out_lon))
    reshaped_3d = reshaped[np.newaxis,:, :]
    result_array = reshaped_sf * reshaped_3d
    result = result_array.flatten().tolist()

    return result,reshaped_sf

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

    #Get LIS data
    lis_flashes_trmm, lis_radiances_trmm, lis_fl_hrly_trmm, lis_rad_hrly_trmm, lis_lat_trmm, lis_lon_trmm = get_lis_data(filename_trmm)
    lis_flashes_iss, lis_radiances_iss, lis_fl_hrly_iss, lis_rad_hrly_iss, lis_lat_iss, lis_lon_iss = get_lis_data(filename_iss)
    lis_flashes_all, lis_radiances_all, lis_fl_hrly_all, lis_rad_hrly_all = get_lis_data_all(filename_trmm,filename_iss)

    #Interpolate 24h radiance data
    lis_radiances_trmm_int = interpolate_24h(lis_radiances_trmm)
    lis_radiances_iss_int = interpolate_24h(lis_radiances_iss)
    lis_radiances_all_int = interpolate_24h(lis_radiances_all)

    #Interpolate hourly radiance data
    lis_rad_hrly_trmm_int = interpolate_hr(lis_rad_hrly_trmm)
    lis_rad_hrly_iss_int = interpolate_hr(lis_rad_hrly_iss)
    lis_rad_hrly_all_int = interpolate_hr(lis_rad_hrly_all)

    #Get beta factor
    scale_factor_trmm = get_conversion_factor(tc_nox_yield, av_no, gm_lpnox, lis_flashes_trmm, lis_radiances_trmm_int)
    scale_factor_iss = get_conversion_factor(tc_nox_yield, av_no, gm_lpnox, lis_flashes_iss, lis_radiances_iss_int)
    scale_factor_all = get_conversion_factor(tc_nox_yield, av_no, gm_lpnox, lis_flashes_all, lis_radiances_all_int)

    #Calculate production rates
    prod_lnox_trmm = get_lnox_prod(tc_nox_yield, scale_factor_trmm, av_no, lis_radiances_trmm_int, lis_flashes_trmm)
    prod_lnox_iss = get_lnox_prod(tc_nox_yield, scale_factor_iss, av_no, lis_radiances_iss_int, lis_flashes_iss)
    prod_lnox_all = get_lnox_prod(tc_nox_yield, scale_factor_all, av_no, lis_radiances_all_int, lis_flashes_all)

    #Get hourly scaling factors
    diel_scale_factors_iss = get_diel_scaling(lis_radiances_iss_int, lis_rad_hrly_iss_int)
    diel_scale_factors_trmm = get_diel_scaling(lis_radiances_trmm_int, lis_rad_hrly_trmm_int)
    diel_scale_factors_all = get_diel_scaling(lis_radiances_all_int, lis_rad_hrly_all_int)

    #Check to see that scaling factors sum to 1 in each grid square
    # Sum over the time dimension
    data_sum = diel_scale_factors_all.sum(axis=2)  # shape (576, 201)
    non_nan_mask = ~np.isnan(data_sum)
    non_nan_values = data_sum[non_nan_mask]
    print("Number of non-NaN grid points:", non_nan_values.size)
    print("Min:", np.min(non_nan_values))
    print("Max:", np.max(non_nan_values))
    print("Mean:", np.mean(non_nan_values))

    #Get 24h production rates
    df_trmm = get_24h_prodrate(lis_lon_trmm,lis_lat_trmm,prod_lnox_trmm)
    df_iss = get_24h_prodrate(lis_lon_iss,lis_lat_iss,prod_lnox_iss)
    df_all = get_24h_prodrate(lis_lon_iss,lis_lat_iss,prod_lnox_all)

    #Grid 24h production rates to GEOS-Chem input grid
    trmm_lnox_new,trmm_lon_new,trmm_lat_new,xdim,ydim,out_lon,out_lat = gridding(df_trmm['lon'],df_trmm['lat'],df_trmm['lnox'],lon_res,lat_res)
    iss_lnox_new,iss_lon_new,iss_lat_new,xdim,ydim,out_lon,out_lat = gridding(df_iss['lon'],df_iss['lat'],df_iss['lnox'],lon_res,lat_res)
    all_lnox_new,all_lon_new,all_lat_new,xdim,ydim,out_lon,out_lat = gridding(df_all['lon'],df_all['lat'],df_all['lnox'],lon_res,lat_res)

    #Get filled 24h production rates
    df_trmm_new = get_filled_24h(trmm_lon_new,trmm_lat_new,trmm_lnox_new)
    df_iss_new = get_filled_24h(iss_lon_new,iss_lat_new,iss_lnox_new)
    df_all_new = get_filled_24h(all_lon_new,all_lat_new,all_lnox_new)

    #Get filled hourly scaling factors
    full_iss = get_hr_sf(lis_lon_trmm,lis_lat_trmm,diel_scale_factors_trmm)
    full_trmm = get_hr_sf(lis_lon_iss,lis_lat_iss,diel_scale_factors_iss)
    full_all = get_hr_sf(lis_lon_iss,lis_lat_iss,diel_scale_factors_all)

    #Confirming all hourly scaling factors sum to 1 in each grid square
    #Sum scaling factors over all times for each grid cell
    df_sum = full_all.groupby(['lat', 'lon'])['lnox'].sum().reset_index()
    #Create pivot table for plotting
    pivot = df_sum.pivot(index='lat', columns='lon', values='lnox')
    #Sort latitudes in ascending order (optional, for correct map orientation)
    pivot = pivot.sort_index()
    #Create the pcolormesh plot
    plt.figure(figsize=(12,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    lats = pivot.index.values
    lons = pivot.columns.values
    data = pivot.values
    mesh = ax.pcolormesh(lons, lats, data, shading='auto', cmap='viridis', vmin=0.9, vmax=1.1)
    ax.coastlines()
    plt.colorbar(mesh, label='Sum of scaling factors', ax=ax)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Sum of scaling factors over all hours')
    plt.show()

    #Apply scaling factors to production rates
    result_trmm,sf_trmm = get_hr_prodrate(trmm_lon_new,trmm_lat_new,trmm_lnox_new,full_trmm['lnox'].tolist(),full_trmm['lon'].tolist(),full_trmm['lat'].tolist(),full_trmm['time'].tolist())
    result_iss,sf_iss = get_hr_prodrate(iss_lon_new,iss_lat_new,iss_lnox_new,full_iss['lnox'].tolist(),full_iss['lon'].tolist(),full_iss['lat'].tolist(),full_iss['time'].tolist())
    result_all,sf_all = get_hr_prodrate(all_lon_new,all_lat_new,all_lnox_new,full_all['lnox'].tolist(),full_all['lon'].tolist(),full_all['lat'].tolist(),full_all['time'].tolist())

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
    lnox_prod_trmm[:] = (df_trmm_new['lnox']*av_no).to_numpy().reshape(1,ydim,xdim)
    # #lnox production from ISS
    lnox_prod_iss = ncout.createVariable('PROD_LNOX_ISS', np.float32, ('time' ,'lat', 'lon'), chunksizes=(1,ydim,xdim))
    lnox_prod_iss.units = '1'
    lnox_prod_iss.long_name = 'Production rate of lightning NOx from ISS'
    lnox_prod_iss[:] = (df_iss_new['lnox']*av_no).to_numpy().reshape(1,ydim,xdim)
    # #lnox production from both
    lnox_prod_all = ncout.createVariable('PROD_LNOX_ALL', np.float32, ('time' ,'lat', 'lon'), chunksizes=(1,ydim,xdim))
    lnox_prod_all.units = '1'
    lnox_prod_all.long_name = 'Production rate of lightning NOx from both TRMM and ISS'
    lnox_prod_all[:] = (df_all_new['lnox']*av_no).to_numpy().reshape(1,ydim,xdim)
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
    lnox_prod_iss[:] = (np.array(result_iss)*av_no).reshape(zdim,ydim,xdim)
    #lnox production from TRMM
    lnox_prod_trmm = ncout.createVariable('PROD_LNOX_TRMM', np.float32, ('time', 'lat', 'lon'), chunksizes=(zdim,ydim,xdim))
    lnox_prod_trmm.units = '1'
    lnox_prod_trmm.long_name = 'Diurnal production rate of lightning NOx from TRMM'
    lnox_prod_trmm[:] = (np.array(result_trmm)*av_no).reshape(zdim,ydim,xdim)
    #lnox production from both
    lnox_prod_all = ncout.createVariable('PROD_LNOX_ALL', np.float32, ('time', 'lat', 'lon'), chunksizes=(zdim,ydim,xdim))
    lnox_prod_all.units = '1'
    lnox_prod_all.long_name = 'Diurnal production rate of lightning NOx from both TRMM and ISS'
    lnox_prod_all[:] = (np.array(result_all)*av_no).reshape(zdim,ydim,xdim)
    # #scaling factors
    lnox_sf_all = ncout.createVariable('SCALE_FACTOR_ALL', np.float32, ('time', 'lat', 'lon'), chunksizes=(zdim,ydim,xdim))
    lnox_sf_all.units = '1'
    lnox_sf_all.long_name = 'Diurnal scaling factors using TRMM and ISS'
    lnox_sf_all[:] = ((full_all['lnox']).to_numpy()).reshape(zdim,ydim,xdim)
    lnox_sf_iss = ncout.createVariable('SCALE_FACTOR_ISS', np.float32, ('time', 'lat', 'lon'), chunksizes=(zdim,ydim,xdim))
    lnox_sf_iss.units = '1'
    lnox_sf_iss.long_name = 'Diurnal scaling factors using ISS'
    lnox_sf_iss[:] = ((full_iss['lnox']).to_numpy()).reshape(zdim,ydim,xdim)
    lnox_sf_trmm = ncout.createVariable('SCALE_FACTOR_TRMM', np.float32, ('time', 'lat', 'lon'), chunksizes=(zdim,ydim,xdim))
    lnox_sf_trmm.units = '1'
    lnox_sf_trmm.long_name = 'Diurnal scaling factors using TRMM'
    lnox_sf_trmm[:] = ((full_trmm['lnox']).to_numpy()).reshape(zdim,ydim,xdim)
    ncout.close()