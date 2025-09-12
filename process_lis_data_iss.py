#%%
#!/usr/bin/env python

import numpy as np
#import pandas as pd
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

def get_output_grid(lat_res,lon_res,region='tropics'):
    '''Define grid to map flash data. Tropics specified, as LIS sensor only detects
       lightning in the tropics'''
    
    if region=='tropics':

        # Output grid resolution the same as GEOS-Chem.
        # The latitude centre (-50 deg and +50 deg) is identical to the
        # GEOS-Chem grid.
        #grid_lon = np.arange(-180+lon_res/2, 180, lon_res) 
        #grid_lat = np.arange(-50+lat_res/2, 50, lat_res)
        grid_lon = np.arange(-180, 180, lon_res) 
        grid_lat = np.arange(-50, 50+lat_res, lat_res)

    return grid_lat, grid_lon

def get_trmm_files(trmm_dir):
    ''' Routine to read TRMM LIS lightning data provided in NetCDF format'''

    # Print statement to track progress:
    print('Reading TRMM LIS NetCDF data')

    # Get and sort files for this year:
    files = sorted(glob.glob( os.path.join( trmm_dir, '*.nc' ) ))

    return files

def get_iss_files(iss_dir,year):
    ''' Routine to read ISS LIS lightning data provided in NetCDF format'''

    # More steps than TRMM, due to the way the data are received from the NASA
    # EarthData portal.

    # Empty arrays"
    files = []
    iss_day = []
    day_count = 0.0

    # Define day in each month of the year
    days_in_mon = [31,28,31,30,31,30,31,31,30,31,30,31]
    # Account for leap years:
    if year==2004 or year==2008 or year==2012 or year==2016 or year==2020:
        days_in_mon[1] = 29

    # Get files for each month:
    for m in range(12):

        strmon = str(m+1)
        if m+1 < 10: strmon = '0'+strmon

        for d in range(days_in_mon[m]):

            strday = str(d+1)
            if d+1 < 10: strday = '0'+strday

            files_for_day = glob.glob( os.path.join( iss_dir, (strmon+strday),
                                                     '*.nc' ) )

            day_count += 1

            if len(files_for_day)>0:

                #print(len(files_for_day))
                tday_of_yr = np.zeros(len(files_for_day))
                tday_of_yr[:] = day_count

                files.extend( files_for_day )
                iss_day.extend( tday_of_yr )

    # Error check:
    print('ISS day count: ', day_count)

    return files, iss_day

def get_lis_de():
    '''Get hourly detection efficiency from Cecil et al. (2014), 
       https://doi.org/10.1016/j.atmosres.2012.06.028.'''

    # Detection limit array as reported in Cecil et al. (2014):
    lis_de = [0.88,0.879994,0.879999,0.880003,0.879996,0.876248,0.849902,0.812438,
              0.763066,0.737929,0.712149,0.692533,0.695175,0.71421,0.73427,0.758117,
              0.802531,0.843822,0.87547,0.879903,0.879999,0.879999,0.880001,0.879995]
    lis_de = np.array(lis_de)

    return lis_de

def save_ncdf(lis_type, lats, lons, lis_rad_24h, lis_fl_24h, n_24h, lis_rad_hr, lis_fl_hr, n_hr, yr1, yr2):

    # Define output file:
    out_dir = os.path.join('/home', 'ucfarh0', 'python', 'lnox', 'Data')
    #out_file = os.path.join( out_dir, lis_type.lower() + '_radiances_annual_mean_' + yr1 + '-' + yr2 + '_025x03125.nc' )
    out_file = os.path.join( out_dir, 'test_iss_lst_new.nc' )

    # Define dimensions:
    nlons = len(lons)
    nlats = len(lats)
    nhrs  = len(lis_rad_hr[0,0,:])

    # Declare output object:
    ncout = Dataset(out_file, mode='w', format='NETCDF4')

    # Define dimensions:
    ncout.createDimension('lat', nlats)
    ncout.createDimension('lon', nlons)
    ncout.createDimension('hours', 24)

    # Define each output variable in turn. Could be made more efficient
    # by setting up a for loop, but given there are only a few variables
    # in the file, this approach is fine.

    # Latitude:
    lat = ncout.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees east'
    lat.long_name = 'latitude'
    lat.short_name = 'lat'
    lat[:] = lats

    # Longitude:
    lon = ncout.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees north'
    lon.long_name = 'longitude'
    lon.short_name = 'lon'
    lon[:] = lons

    # 24-hour mean radiances:
    rad_24h = ncout.createVariable('rad_24h', np.float32, ('lon','lat'))
    rad_24h.units = 'uJ/sr/m^2/um'
    rad_24h.long_name = 'LIS 24h mean radiances'
    rad_24h.short_name = 'LIS 24h rad'
    rad_24h[:] = lis_rad_24h

    # 24-h total flash counts:
    fl_24h = ncout.createVariable('fl_24h', np.float32, ('lon','lat'))
    fl_24h.units = 'unitless'
    fl_24h.long_name = 'LIS 24h number of flashes'
    fl_24h.short_name = 'LIS 24h fl'
    fl_24h[:] = lis_fl_24h

    # Number of 24-hour data points:
    nobs_24h = ncout.createVariable('nobs_24h', np.float32, ('lon','lat'))
    nobs_24h.units = 'unitless'
    nobs_24h.long_name = 'LIS 24h number of flash observations'
    nobs_24h.short_name = 'LIS 24h nobs'
    nobs_24h[:] = n_24h

    # Hourly mean radiances:
    rad_hr = ncout.createVariable('rad_hr', np.float32, ('lon','lat','hours'))
    rad_hr.units = 'uJ/sr/m^2/um'
    rad_hr.long_name = 'LIS UTC hourly mean radiances'
    rad_hr.short_name = 'LIS hr rad'
    rad_hr[:] = lis_rad_hr

    # Total flashes in each hour:
    fl_hr = ncout.createVariable('fl_hr', np.float32, ('lon','lat','hours'))
    fl_hr.units = 'uJ/sr/m^2/um'
    fl_hr.long_name = 'LIS UTC hourly mean radiances'
    fl_hr.short_name = 'LIS hr rad'
    fl_hr[:] = lis_fl_hr

    # Total number of LIS observations in each hour:
    nobs_hr = ncout.createVariable('nobs_hr', np.float32, ('lon','lat','hours'))
    nobs_hr.units = 'unitless'
    nobs_hr.long_name = 'LIS UTC hourly number of flash observations'
    nobs_hr.short_name = 'LIS hr nobs'
    nobs_hr[:] = n_hr

    ncout.close()

def plot_data(lon, lat, plot_rads, plot_fl):

    # 2D lat-long grid:
    X, Y = np.meshgrid( lon, lat, indexing='ij' )

    # Plot frame:
    fig, ax = plt.subplots(2, 1, figsize=(9, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))

    data_crs = ccrs.PlateCarree()

    # Loop over data plotting:
    for nn in range(2):

        if nn==0:
            plot_vals = plot_rads
            max_val = 1.5
            cbar_title = 'LIS radiance in 2001 [J/sr/m^2/um]'
        if nn==1:
            plot_vals = plot_fl
            max_val = 200
            cbar_title = 'Number of flashes'

        ax[nn].coastlines(resolution='50m', linewidth=0.7)
        ax[nn].set_extent([min(out_lon),max(out_lon),min(out_lat),max(out_lat)],
                          crs=ccrs.PlateCarree())

        c = ax[nn].pcolormesh(X, Y, plot_vals, transform=data_crs,
                              cmap='rainbow', vmin=0, vmax=max_val)

        cb = fig.colorbar(c, ax=ax[nn], label=cbar_title,
                          orientation='horizontal',
                          shrink=0.35,pad=0.03)
            
        cb.ax.tick_params(labelsize=10, direction='in', length=9)
    
    plt.show()


if __name__ == "__main__":

    ''' To process LIS data, need to define years to process below. Comment clarifies
        which years to process for the TRMM sensor and which to process for the ISS
        sensor.

        This code is slow, but can be run in the background. It reads in and processes 
        many LIS NetCDF files for many years.

        The code yields a NetCDF file of the processed data to use to calculate lightning
        NOx per flash production rates. File path in save_ncdf needs to be updated to your
        local directory to save the file.'''

    # Define years:
    # 2002-2014 for TRMM, as orbit altitude boosted in 2001 and decommissioning started
    # in 2015.
    # 03/2017 to 07/2023 for LIS.
    startyy = 2017
    endyy   = 2023
    years = np.arange(startyy, endyy+1, 1)

    # Define final grid:
    #out_lat, out_lon = get_output_grid(0.25,0.3125,region='tropics')
    out_lat, out_lon = get_output_grid(0.5,0.625,region='tropics')
    imx = len(out_lon)
    jmx = len(out_lat)

    # Define hours:
    out_hr = np.arange(0.5, 24, 1)
    out_hr = out_hr
    nhrs = len(out_hr)

    # Define output grid:
    # Annual 24-h mean data:
    grid_24h_rad   = np.zeros((imx,jmx))
    grid_24h_fl    = np.zeros((imx,jmx))
    grid_24h_count = np.zeros((imx,jmx))
    # Annual hourly mean data:    
    grid_hr_rad   = np.zeros((imx,jmx,nhrs))
    grid_hr_fl    = np.zeros((imx,jmx,nhrs))
    grid_hr_count = np.zeros((imx,jmx,nhrs))

    # Define root directory:
    root_dir = '/shared/ucl/depts/uptrop/Data/Satellites/LIS/'

    # Days in each year from 1993 to 2023 to calculate time:
    days_in_yr = [365,365,365,366,365,365,365,366,365,365,
                  365,366,365,365,365,366,365,365,365,366,
                  365,365,365,366,365,365,365,366,365,365,365]

    # Years from 1993 to 2023 to calculate time:
    years_from_1993 = np.arange(1993, 2024, 1)
    
    # Define months:
    months = np.arange(1,12,1)

    lis_de = get_lis_de()
    
    # Select which data type to read in based on year
    # TRMM data on Myriad is for 2001-2015 and ISS data on Myriad is for
    # 2017-2023.

    # Process TRMM data from 2002 to 2015. Exclude years prior to 2002 due to possible
    # complications from TRMM orbital boos in August 2001 and years after 2015 due to inter-
    # mittancy of TRMM before being decommissioned.
    # This follows the same approach as Chronis and Koshak, BAMS, 2017
    # (https://journals.ametsoc.org/view/journals/bams/98/7/bams-d-16-0041.1.xml?tab_body=pdf).

    # Subdirectories for TRMM are ./YYYY/
    # Subdirectories for ISS are ./YYYY/MMDD/

    # File name convention for TRMM is TRMM_LIS_SC.04.x_YYYY.day*, where
    # x is the version number and day is from 001 to 365/366.

    # File name convection for ISS is ISS_LIS_SC_Vx.x_YYYYMMDD_*, where
    # x.x is the version number (either 2.1 or 2.2).

    # Loop over years:
    for yr in years:

        # Initialize:
        total_fl    = 0.0
        fatal_fl    = 0.0
        nonfatal_fl = 0.0

        print('Processing year ' + str(yr))

        # Find index for this year:
        yr_ind = np.where(years_from_1993 == yr)[0]
        # Get days since 1993:
        days_since_1993 = np.sum(days_in_yr[0:yr_ind[0]])
        # Get seconds since 1993 for the start of this year:
        sec_since_1Jan93 = days_since_1993 * 24 * 60 * 60

        # Track files with no detected flashes:
        empty_files = 0

        # Number of days in the year:
        ndays = 365
        # Account for leap years:
        if yr==2004 or yr==2008 or yr==2012 or yr==2016 or yr==2020:
            ndays = 366

        # Define path:
        if yr < 2016:
            # TRMM:
            lis_type = 'TRMM'
            dir_path = os.path.join(root_dir, 'TRMM', str(yr))
            # Get files:
            lis_files = get_trmm_files(dir_path)
        elif yr > 2016:
            # ISS:
            lis_type = 'ISS'
            dir_path = os.path.join(root_dir, 'ISS', str(yr))
            lis_files, files_day_of_yr = get_iss_files(dir_path,yr)
        else:
            # No data for these years:
            continue

        # Track progress:
        print('Found ' + str(len(lis_files)) + ' files')

        # Loop over files:
        for i,f in enumerate(lis_files):
            # Get day of year from filename:
            if lis_type == 'TRMM':
                f_rev = f[::-1]
                day_of_yr = f_rev[9:12]
                day_of_yr = int(day_of_yr[::-1])

            if lis_type == 'ISS':
                day_of_yr = files_day_of_yr[i]

            df = Dataset( f, mode='r' )

            # Check file has flash data:
            try:
                # Flash lat [deg]:
                fl_lat = df.variables['lightning_flash_lat'][:]
                if lis_type=='TRMM': file_segment = f[55:89]
                if lis_type=='ISS': file_segment = f[59:97]
                print('Found flash data in file ' + file_segment + '.')
            except:
                if lis_type=='TRMM': file_segment = f[55:89]
                if lis_type=='ISS': file_segment = f[59:97]
                print('No flash data in file ' + file_segment + '. Skipping!')
                empty_files += 1
                continue

            # Extract relevant data:
            # Flash lat [deg]:
            fl_lat = df.variables['lightning_flash_lat'][:]
            # Flash lon [deg]:
            fl_lon = df.variables['lightning_flash_lon'][:]
            # Flash alert flag (use only if value is 8?) [unitless]:
            # Flag numbers:
            # 0: none
            # 1: instrument_fatal
            # 2: instrument_warning
            # 4: platform_fatal
            # 8: platform_warning
            # 16: external_fatal
            # 32: external_warning
            # 64: processing_fatal
            #128: processing_warning
            fl_flag = df.variables['lightning_flash_alert_flag'][:]
            # Flash footprint [km^2]:
            fl_footprint = df.variables['lightning_flash_footprint'][:]
            # Flash radiance [uJ/sr/m^2/um]:
            fl_radiance = df.variables['lightning_flash_radiance'][:]
            # Flash time (presumably UTC?) [sec since 1993-01-01 00:00:00.000]:
            fl_time = df.variables['lightning_flash_TAI93_time'][:]
            # Flash duration (delta_time) [sec]:
            fl_duration = df.variables['lightning_flash_delta_time'][:]

            df.close()

            # threshold = 1e7
            # # Flag to check if any radiance value exceeds threshold
            # radiance_exceeds_threshold = False
            # for num in fl_radiance:
            #     if num > threshold:
            #         radiance_exceeds_threshold = True
            #         print(f"Skipping file {f} because it has a radiance greater than the threshold ({threshold}).")
            #         break  # Break the loop as soon as threshold exceeded
            # if radiance_exceeds_threshold:
            #     continue

            print(f"Processing file {f}")

            # Get UTC hour of flash detection:
            sec_since_93 = sec_since_1Jan93 + ((day_of_yr-1)*24*60*60)
            utc_hr = np.array(fl_time-sec_since_93)/(60*60)

            # Loop over LIS data:
            for i in np.arange(len(fl_lat)):

                bin_fl_flag = "{:08b}".format(fl_flag[i])

                total_fl += 1

                tlon = fl_lon[i]

                # Find data flagged as fatal and count number of fatal and non-fatal
                # flagged data points:
                if ((bin_fl_flag[7]=='1') or (bin_fl_flag[5]=='1') or
                    (bin_fl_flag[3]=='1') or (bin_fl_flag[1]=='1')):
                    fatal_fl += 1
                    continue
                else:
                    nonfatal_fl += 1

                # Longitude correction:
                if fl_lon[i] < 0:
                    tlon = fl_lon[i] + 360

                # UTC time correction:
                if utc_hr[i] > 24:
                    tutc_hr = utc_hr[i] - 24
                elif utc_hr[i] < 0:
                    #print('Negative UTC hour: ',utc_hr[i])
                    tutc_hr = utc_hr[i] + 24                    
                else:
                    tutc_hr = utc_hr[i]

                # Get local solar time from longitude and UTC hour:
                lst_hr = tutc_hr + (tlon / 15)

                # Local solar time correction:
                if lst_hr > 24:
                    lst_hr = lst_hr - 24
                if lst_hr < 0:
                    #print('LST,UTC: ', lst_hr,tutc_hr,tlon,lst_hr+24)
                    lst_hr = lst_hr + 24
            
                # Find target grid indices:
                lon_ind = np.argmin(np.abs(fl_lon[i] - out_lon))
                lat_ind = np.argmin(np.abs(fl_lat[i] - out_lat))

                # Find target hour:
                #hr_ind = np.argmin(np.abs(tutc_hr - out_hr))
                hr_ind = np.argmin(np.abs(lst_hr - out_hr))


                # Error check:
                if hr_ind==23:
                    if lst_hr > 24:
                        #print(lst_hr, utc_hr[i], fl_lon[i])
                        #print('Local solar time > 24')
                        sys.exit()

                # Error check:
                if lst_hr < 0:
                    #print(lst_hr, utc_hr[i], fl_lon[i])
                    #print('Negative local solar time')
                    sys.exit()

                # Apply time-dependent detection efficiency correction from
                # Cecil et al. (2014) ( https://doi.org/10.1016/j.atmosres.2012.06.028)
                # and sum up data to fixed lat and long grid:
                # (1) To calculate 24h annual mean:

                if fl_radiance[i] == 0:
                    grid_24h_rad[lon_ind, lat_ind] += np.nan #/ lis_de[hr_ind]
                    grid_24h_fl[lon_ind, lat_ind] += 1.0 / lis_de[hr_ind]
                    grid_24h_count[lon_ind, lat_ind] += 1.0
                if fl_radiance[i] > 1e7:
                    # Do not add to lists
                    pass
                else:
                    grid_24h_rad[lon_ind, lat_ind] += fl_radiance[i] / lis_de[hr_ind]
                    grid_24h_fl[lon_ind, lat_ind] += 1.0 / lis_de[hr_ind]
                    grid_24h_count[lon_ind, lat_ind] += 1.0
                # (2) To calculate hourly annual mean:
                if fl_radiance[i] == 0:
                    grid_hr_rad[lon_ind,lat_ind,hr_ind] += np.nan #/ lis_de[hr_ind]
                    grid_hr_fl[lon_ind,lat_ind,hr_ind] += 1.0 / lis_de[hr_ind]
                    grid_hr_count[lon_ind,lat_ind,hr_ind] += 1.0
                if fl_radiance[i] > 1e7:
                    # Do not add to lists
                    pass
                else:
                    grid_hr_rad[lon_ind,lat_ind,hr_ind] += fl_radiance[i] / lis_de[hr_ind]
                    grid_hr_fl[lon_ind,lat_ind,hr_ind] += 1.0 / lis_de[hr_ind]
                    grid_hr_count[lon_ind,lat_ind,hr_ind] += 1.0
        # Print statements to track files with no flashes and
        # the number of flashes flagged as fatal.
        print('No. empty files in ' + str(yr) + ' = ' + str(empty_files))
        print(str(total_fl) + ' total ' + str(fatal_fl) + ' fatal ' +
              str(nonfatal_fl) + ' nonfatal')

    # Get grid average radiances based on number of detected flashes rather
    # than flashes, as flashes are corrected for detection efficiencies:
    iind,jind = np.where(grid_24h_count > 10)
    # print('Min 24h grid count: ', np.min(grid_24h_count[iind,jind]))
    # print('Max 24h grid count: ', np.max(grid_24h_count[iind,jind]))
    # print('Mean 24h grid count: ', np.mean(grid_24h_count[iind,jind]))
    #if grid_24h_count[iind,jind] >= 5:
    grid_24h_rad[iind,jind] = ( grid_24h_rad[iind,jind] /
                                grid_24h_count[iind,jind] )
    # else:
    #     grid_24h_rad[iind,jind] = np.nan
    #     grid_24h_count[iind,jind] = np.nan
    
    iind,jind,hind = np.where(grid_hr_count > 10)
    # print('Min hr grid count: ', np.min(grid_hr_count[iind,jind,hind]))
    # print('Max hr grid count: ', np.max(grid_hr_count[iind,jind,hind]))
    # print('Mean hr grid count: ', np.mean(grid_hr_count[iind,jind,hind]))
    #if len(grid_hr_count[iind,jind,hind]) >= 5:
    grid_hr_rad[iind,jind,hind] = ( grid_hr_rad[iind,jind,hind] /
                                    grid_hr_count[iind,jind,hind] )
    # else:
    #     grid_hr_rad[iind,jind,hind] = np.nan
    #     grid_hr_count[iind,jind,hind] = np.nan

    
    # print('Grid rad: ', grid_hr_rad[iind,jind,hind])
    # print('Grid count: ', grid_hr_count[iind,jind,hind])
    # Print statement to track / check that gridding gives reasonable values:            
    print('Mean radiance: ', np.mean(grid_hr_rad[iind,jind,hind]*1e-6))
    print('Max grid count: ', np.max(grid_hr_count))

    # Save the data:
    save_ncdf(lis_type, out_lat, out_lon, grid_24h_rad, grid_24h_fl, grid_24h_count, grid_hr_rad, grid_hr_fl, grid_hr_count, str(startyy), str(endyy))

    # Plot the data:
    plot_data(out_lon, out_lat, grid_24h_rad*1e-6, grid_24h_fl)

    sys.exit()

# %%
