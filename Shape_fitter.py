import numpy as np
import scipy
import matplotlib.pyplot as plt 
import datetime as dt
import pandas as pd
from scipy import signal
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import pickle
import matplotlib.backends.backend_pdf

def import_text(creepmeter):
    ''' import text file for creepmeter data
    input creepmter: creepmeter name
    return tm/tm2: time for data
    return min10creep/min10_creep2: slip for data'''
    if creepmeter == 'XSJ' or creepmeter == 'XHR' or creepmeter == 'XPK':
        vls = np.loadtxt("../../DATA_10MIN/RAW/San_Andreas/Tidy_name/{K}_A_10min.txt".format(K=creepmeter), dtype = str)
        Year  = vls[:,0].astype(int)
        Time  = vls[:,1].astype(float)
        min10_creep  = vls[:,2].astype(float)
        tm =np.array([dt.datetime(Year[k],1,1) + dt.timedelta(days = Time[k] -1) for k in range (0, len(Year))])

        vls2 = np.loadtxt("../../DATA_10MIN/RAW/San_Andreas/Tidy_name/{K}_B_10min.txt".format(K=creepmeter), dtype = str)
        Year2  = vls2[:,0].astype(int)
        Time2  = vls2[:,1].astype(float)
        min10_creep2  = vls2[:,2].astype(float)
        tm2 =np.array([dt.datetime(Year2[k],1,1) + dt.timedelta(days = Time2[k] -1) for k in range (0, len(Year2))])
    elif creepmeter == 'XMR':
        vls = np.loadtxt("../../DATA_10MIN/RAW/San_Andreas/Tidy_name/{K}_10min.txt".format(K=creepmeter), dtype = str)
        Year  = vls[:,0].astype(int)
        Time  = vls[:,1].astype(float)
        min10_creep_ALL  = vls[:,2].astype(float)
        tm_ALL =np.array([dt.datetime(Year[k],1,1) + dt.timedelta(days = Time[k] -1) for k in range (0, len(Year))])
        boolarr10 = (tm_ALL > dt.datetime(1991,1,1,0,0,0))&(tm_ALL< dt.datetime(2017,1,1,0,0,0))
        tm = tm_ALL[boolarr10]
        min10_creep = min10_creep_ALL[boolarr10]
        boolarr1 = (tm_ALL> dt.datetime(2017,1,1,0,0,0))
        tm2=tm_ALL[boolarr1]
        min10_creep2=min10_creep_ALL[boolarr1]
    else:
        vls = np.loadtxt("../../DATA_10MIN/RAW/San_Andreas/Tidy_name/{K}_10min.txt".format(K=creepmeter), dtype = str)
        Year  = vls[:,0].astype(int)
        Time  = vls[:,1].astype(float)
        min10_creep  = vls[:,2].astype(float)
        tm =np.array([dt.datetime(Year[k],1,1) + dt.timedelta(days = Time[k] -1) for k in range (0, len(Year))])
        tm2=()
        min10_creep2=()
    return tm, min10_creep, tm2, min10_creep2



def interpolate(tm,min10_creep,creepmeter):
    '''interpolate the time series data to 10 minute frequency
        input               tm: time
        input      min10_creep: slip
        return          tm_int: interpolated time
        return min10_creep_int: interpolated slip'''
    
    Time = pd.Series(pd.to_datetime(tm)) #convert to pandas series
    creeping = pd.DataFrame({'Time':Time, 'Tm': Time,'Creep':min10_creep.astype(np.float)}) #create a pandas dataframe
    creeping.Time = creeping.Time.dt.round("10min") #round creep times to nearest 10 mins (make evenly spaced)
    creeping.Tm = creeping.Tm.dt.round("10min")
    creeping.set_index('Time',inplace=True) #set index of the dataframe
    creeping.drop_duplicates(subset=['Tm'], inplace=True) 
    upsampled = creeping.resample('10min').ffill(1) #upsample the timeframe to get a uniformly spaced dataset
    upsampled['Time'] = upsampled.index #get time as a column
    interpolated = upsampled.interpolate(method = 'ffill') #interpolate the dataset to get a continious record evenly spaced at 10 mins
    tm_int = np.array(interpolated.Time) #make Time and creep into Numpy array.
    min10_creep_int = np.array(interpolated.Creep)
    return tm_int, min10_creep_int

def interpolate_1min(tm,min10_creep,creepmeter):
    '''interpolate the time series data to 10 minute frequency
        input               tm: time
        input      min10_creep: slip
        return          tm_int: interpolated time
        return min10_creep_int: interpolated slip'''
    
    Time = pd.Series(pd.to_datetime(tm)) #convert to pandas series
    creeping = pd.DataFrame({'Time':Time, 'Tm': Time,'Creep':min10_creep.astype(np.float)}) #create a pandas dataframe
    creeping.Time = creeping.Time.dt.round("1min") #round creep times to nearest 10 mins (make evenly spaced)
    creeping.Tm = creeping.Tm.dt.round("1min")
    creeping.set_index('Time',inplace=True) #set index of the dataframe
    creeping.drop_duplicates(subset=['Tm'], inplace=True) 
    upsampled = creeping.resample('1min').ffill(1) #upsample the timeframe to get a uniformly spaced dataset
    upsampled['Time'] = upsampled.index #get time as a column
    interpolated = upsampled.interpolate(method = 'ffill') #interpolate the dataset to get a continious record evenly spaced at 10 mins
    tm_int = np.array(interpolated.Time) #make Time and creep into Numpy array.
    min10_creep_int = np.array(interpolated.Creep)

    return tm_int, min10_creep_int



def creepmeter_events(creepmeter):
    '''import creep catalogue for creep event
    input creepmeter: name of creepmeter
    return df_PICKS: dataframe of creep events
    return duration: numpy array of duration of each event'''
    df_PICKS = pd.read_csv('../../CREEP_CATALOGUE/Creep event catalog at {k}.csv'.format(k=creepmeter),index_col=0)
    df_PICKS['og_index'] = df_PICKS.index #get creep event number not in index column
    EVENTS = df_PICKS.og_index
    #extract start and end times for the creep events
    
    START_1 = pd.to_datetime(pd.Series(df_PICKS.Start_Time))
    END_1 = pd.to_datetime(pd.Series(df_PICKS.End_Time))
    if creepmeter == 'XMD':
        END_1.iloc[0] = END_1.iloc[0]-dt.timedelta(minutes=10)
    duration=((END_1-START_1)/dt.timedelta(hours=1))
    median_duration = np.median(duration)
    return df_PICKS, duration,START_1



def vel_acc(Time,slip,dx):
    '''determine the velocity and acceleraton of the timeseries
    input Time: time
    input slip: slip
    input   dx: time difference
    return data: dataframe containing time, slip, velocity and acceleration'''
    Time_new = pd.Series(Time).dt.round("10min") #round times to nearest 10 minutes
    #dx = 10 #time difference in minutes
    #dx =10/60 # time difference in hours
    V = np.gradient(slip,dx)
    A = np.gradient(V,dx) # calculate the acceleration
    data = pd.DataFrame({'Time':Time_new,'Creep':slip,'vel':V,'acc':A})
    return data

def vel_acc_1min(Time,slip,dx):
    '''determine the velocity and acceleraton of the timeseries
    input Time: time
    input slip: slip
    input   dx: time difference
    return data: dataframe containing time, slip, velocity and acceleration'''
    Time_new = pd.Series(Time).dt.round("1min") #round times to nearest 10 minutes
    #dx = 10 #time difference in minutes
    #dx =10/60 # time difference in hours
    V = np.gradient(slip,dx)
    A = np.gradient(V,dx) # calculate the acceleration
    data = pd.DataFrame({'Time':Time_new,'Creep':slip,'vel':V,'acc':A})
    return data

def parkfield_remover(dataframe,creepmeter):
    ''' remove the period of time around the 2004 Parkfield earthquake
    input   dataframe: dataframe containing time, slip, velocity and acceleration
    input  creepmeter: creepmeter invesigating
    return dataframe2: return dataframe with time of Parkfield earthqauke removed. '''
    if creepmeter == 'XMM' or creepmeter == 'XMD' or creepmeter == 'XVA' or creepmeter == 'XPK' or creepmeter == 'XTA' or creepmeter == 'WKR' or creepmeter == 'CRR' or creepmeter == 'XGH':
        idx = np.logical_or(pd.to_datetime(dataframe.Start_Time)<=dt.datetime(2004,9,28,0,0,0),pd.to_datetime(dataframe.Start_Time)>=dt.datetime(2009,9,28,0,0,0))
        Zeros = prop_pos(dataframe,idx)
        dataframe2 = dataframe.copy(deep=True)
        dataframe2['Parkfield'] = Zeros
    else:
        Zeros = np.zeros(len(dataframe))
        dataframe2 = dataframe.copy(deep=True)
        dataframe2['Parkfield']= Zeros
    return dataframe2

def lat_lon(creepmeter):
    '''provide location of creepmeter
    input creepmeter: creepmeter name
    return latlon: lat & lon of creepmeter'''
    XSJ_latlon = {'name': 'XSJ', 'lat': 36.837, 'lon': -121.52}
    XHR_latlon = {'name': 'XHR', 'lat': 36.772 , 'lon': -121.422}
    CWN_latlon = {'name': 'CWN', 'lat': 36.750 , 'lon': -121.385}
    XMR_latlon = {'name': 'XMR', 'lat': 36.595 , 'lon': -121.187}
    XSC_latlon = {'name': 'XSC', 'lat': 36.065, 'lon': -120.628}
    XMM_latlon = {'name': 'XMM', 'lat': 35.958, 'lon': -120.502}
    XMD_latlon = {'name': 'XMD', 'lat': 35.943, 'lon': -120.485}
    XVA_latlon = {'name': 'XVA', 'lat': 35.922, 'lon': -120.462}
    XRSW_latlon = {'name': 'XRSW', 'lat': 35.907, 'lon': -120.46}
    XPK_latlon = {'name': 'XPK', 'lat': 35.902, 'lon': -120.442}
    XTA_latlon = {'name': 'XTA', 'lat': 35.89, 'lon': -120.427}
    XHSW_latlon = {'name': 'XHSW', 'lat': 35.862, 'lon': -120.415}
    WKR_latlon = {'name': 'WKR', 'lat': 35.858, 'lon': -120.392}
    CRR_latlon = {'name': 'CRR', 'lat': 35.835, 'lon': -120.363}
    XGH_latlon = {'name': 'XGH', 'lat': 35.82, 'lon': -120.348}
    C46_latlon = {'name': 'C46', 'lat': 35.730, 'lon': -120.290}
    X46_latlon = {'name': 'X46', 'lat': 35.723, 'lon': -120.278}
    
    latlon = eval('{k}_latlon'.format(k=creepmeter))
    return latlon


def rain_time_series(fname,starttime,endtime,location):
    """
    read ECMWF pressure data and output timeseries
    :param        fname: netCDF file
    :param    starttime: start time of data
    :param      endtime: end time of data
    :param     location: location of strainmeter
    :return     dt_time: time of data
    :return    pressure: pressure data    
    """ 
    import netCDF4 as nc
    #import file
    ds = nc.Dataset(fname)
    
    #for var in ds.variables.values():
    #    print(var)
    
    #extract variables
    lats = ds.variables['latitude'][:]
    lons = ds.variables['longitude'][:]
    time = ds.variables['time'][:]
    pressure_all = ds.variables['tp'][:]
    
    #isolate location of closest grid point to strainmeter
    lat_idx = np.abs(lats - location['lat']).argmin()
    lon_idx = np.abs(lons - location['lon']).argmin()
    
    #create time array for duration of pressure data
    dt_time = np.array(pd.date_range(starttime,endtime,freq='H'))
    
    #extract pressure data
    rainfall = np.array(pressure_all[:, lat_idx, lon_idx])
    
    #return pressure data
    return dt_time, rainfall



def combine_rain(fname1,fname2,fname3,fname4,starttime,endtime,latlon):
    '''combine multiple rainfall records together
    input fname1-4: names of rainfall files
    input starttime: start time of rainfall records
    input endtime: end time of rainfall records
    input latlon: lat & lon of creepmeter
    return dt_time: time of rainfall record
    return rainfall: rainfall recordings'''
    dt_time1, rainfall1  = rain_time_series(fname1,starttime,endtime,latlon)
    dt_time2, rainfall2  = rain_time_series(fname2,starttime,endtime,latlon)
    dt_time3, rainfall3  = rain_time_series(fname3,starttime,endtime,latlon)
    dt_time4, rainfall4  = rain_time_series(fname4,starttime,endtime,latlon)
    
    #combine to one numpy array
    dt_time = np.array(pd.date_range(dt_time1[0],dt_time4[-1],freq='H'))
    rainfall = ()
    for i in range(4):
        rainfall = np.append(rainfall,eval('rainfall{k}'.format(k=i+1)))
    rainfall = rainfall*1000 #put measurements in mm rather than m
    return dt_time , rainfall

def rain_timeseries(creepmeter):
    fname1 = '../../Rainfall/ECMWF/ECMWF_Rainfall_SAF_1985-1989.nc'
    fname2 = '../../Rainfall/ECMWF/ECMWF_Rainfall_SAF_1990-1999.nc'
    fname3 = '../../Rainfall/ECMWF/ECMWF_Rainfall_SAF_2000-2009.nc'
    fname4 = '../../Rainfall/ECMWF/ECMWF_Rainfall_SAF_2010-2020.nc'
    starttime_rain = "1985-JAN-01 00:00:00"
    endtime_rain = "2020-DEC-31 23:00:00"
    
    latlon = lat_lon(creepmeter)
    rainfall_time,rainfall_creepmeter = combine_rain(fname1,fname2,fname3,fname4,starttime_rain,endtime_rain,latlon)
    df_rain = pd.DataFrame({'Time':rainfall_time,'Tm':rainfall_time,'PRCP_creepmeter':rainfall_creepmeter})
    df_rain.set_index('Tm',inplace=True)
    #df_rain_day_total = df_rain.copy(deep=True)
    return df_rain



def rain_finder_general(dataframe_creep, dataframe_rain,time_window):
    '''input dataframe_creep: dataframe of creep events
       input dataframe_rain: dataframe of rain
       input time_window: time for classification as a rain related event in days
       return unique_CM2: dataframe of events not associated with rain'''
    Rain_CM2 = ()
    #dataframe_rain_dropped = dataframe_rain.copy(deep=True)
    dataframe_rain.drop(dataframe_rain[(dataframe_rain['PRCP_creepmeter'] <= 0.1)].index, inplace=True) #can add a threshold here as having it trip with 10^-14 mm of rain seems wrong
    for i in range(len(dataframe_rain)):
        boolarr_RAIN_CM2 = (dataframe_rain['Time'].iloc[i] >= pd.to_datetime(dataframe_creep.Start_Time) - dt.timedelta(days=time_window)) & (dataframe_rain['Time'].iloc[i] <= pd.to_datetime(dataframe_creep.Start_Time)) 
            # create boolian for rain or no rain within a certain window before the creep event
        #print(boolarr_RAIN_CM2)
        for j in range(len(boolarr_RAIN_CM2)):
            if boolarr_RAIN_CM2[j] == True:
                Rain_CM2 = np.append(Rain_CM2,j) #identfy times when rain is before
            else:
                dummy=1
    unique_CM2 = np.unique(Rain_CM2, axis=0)
    
    return unique_CM2


def prop_pos(dataframe,list_prop):
    Zeros = np.zeros(len(dataframe))

    for i in range(len(dataframe)):
        if i in list_prop:
            Zeros[i] = 1
        else:
            dummy=12
    return Zeros

def when_does_it_rain(event_dataframe,creempeter):
    df_rain_day_total = rain_timeseries(creempeter)
    rain_drop = rain_finder_general(event_dataframe, df_rain_day_total,1)
    Zeros = prop_pos(event_dataframe,rain_drop)
    event_dataframe2 = event_dataframe.copy(deep=True)
    event_dataframe2['rain_poss'] = Zeros
    return event_dataframe2


def creep_event_dataframe(dataframe,duration, start, creep_data,creepmeter):
    dataframes={}
    creep_index = np.array(dataframe.og_index)
    for j in range(len(dataframe)):
        Creep_event_time = ()
        Creep_event_slip = ()
        Creep_event_slip_rate = ()
        #boolarr = (START.iloc[j].replace(tzinfo=None)<= data.Time) & (data.Time <= END.iloc[j].replace(tzinfo=None))
        boolarr = (start.iloc[j] <= creep_data.Time) & (creep_data.Time <= start.iloc[j] + dt.timedelta(hours=duration[j])) #assumption made here
        Creep_event_time = creep_data.Time[boolarr]
        Creep_event_time = (Creep_event_time - Creep_event_time.iloc[0])/dt.timedelta(hours=1) #set start time to 0
        Creep_event_slip = creep_data.Creep[boolarr]
        #Creep_event_slip = creep_data.rolling_mean[boolarr] #extract slip
        Creep_event_slip = Creep_event_slip - Creep_event_slip.iloc[0] #set initial slip to 0
        Velocity = creep_data.vel[boolarr]
        Acceleration = creep_data.acc[boolarr]
        dataframes[creep_index[j]] = pd.DataFrame({'Time':Creep_event_time,'Slip':Creep_event_slip,'Velocity':Velocity,'Acceleration':Acceleration})
        dataframes[creep_index[j]].reset_index(inplace=True)
        
        with open("../../Rheology/{k}/dataframe_of_events_at_{k}_01_MAR_22.txt".format(k=creepmeter), "wb") as tf:
            pickle.dump(dataframes,tf)
        with open("../../Rheology/{k}/event_index_{k}_01_MAR_22.txt".format(k=creepmeter),"wb") as tf2:
            pickle.dump(creep_index,tf2)
    return dataframes, creep_index


def VIS_obj(to_opt,OBS_Time,OBS_Data,C_matrix_inv_selection):
    T0 = to_opt[0]
    Tau = to_opt[1]
    V_0 = to_opt[2]
    D0 = to_opt[3]  
    Rheo_guess = D0 + V_0*Tau*(1-np.exp(-(OBS_Time-T0)/Tau))
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - Rheo_guess))
    Error_co_VIS = np.dot(np.array(np.transpose(OBS_Data - Rheo_guess)),BC)
    
    denominator = np.dot(C_matrix_inv_selection,np.array(OBS_Data))
    ratio_denominator = np.dot(np.array(OBS_Data),denominator)
    ratio = Error_co_VIS/ratio_denominator
    return ratio

def DUC_obj(to_opt2,OBS_Time,OBS_Data,C_matrix_inv_selection):
    T0 = to_opt2[0]
    Tau = to_opt2[1]
    V_0 = to_opt2[2]
    D0 = to_opt2[3]
    n = to_opt2[4]
    
    Rheo_guess  = D0 + V_0*Tau*n*(1-(1+(1-1/n)*((OBS_Time-T0)/Tau))**(1/(1-n)))
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - Rheo_guess))
    Error_co_DUC = np.dot(np.array(np.transpose(OBS_Data - Rheo_guess)),BC)
    
    denominator = np.dot(C_matrix_inv_selection,np.array(OBS_Data))
    ratio_denominator = np.dot(np.array(OBS_Data),denominator)
    ratio = Error_co_DUC/ratio_denominator
    return ratio


'''def VSF_obj(to_opt3,OBS_Time,OBS_Data):
    T0 = to_opt3[0]
    Tau = to_opt3[1]
    V = to_opt3[2]
    D0 = to_opt3[3]
    K = to_opt3[4]
    
    Rheo_guess = D0 + K*np.log(V*(np.exp(1)**((t-T0)/Tau)-1)+1)
    BC = np.dot(C_matrix_inv_selection,np.array(Real_data - Rheo_guess))
    Error_co_VSF = np.dot(np.array(np.transpose(Real_data - Rheo_guess)),BC)
    return Error_co_VSF'''


def VSF_obj(to_opt3,OBS_Time,OBS_Data,C_matrix_inv_selection):
    T0 = to_opt3[0]
    Tau = to_opt3[1]
    V_0 = to_opt3[2]
    D0 = to_opt3[3]    
    
    Rheo_guess = D0 + V_0*Tau*np.log(1+((OBS_Time-T0)/Tau))
    
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - Rheo_guess))
    Error_co_VSF = np.dot(np.array(np.transpose(OBS_Data - Rheo_guess)),BC)
    
    denominator = np.dot(C_matrix_inv_selection,np.array(OBS_Data))
    ratio_denominator = np.dot(np.array(OBS_Data),denominator)
    ratio = Error_co_VSF/ratio_denominator
    return ratio


def CB_obj(to_opt4,OBS_Time,OBS_Data,C_matrix_inv_selection):
    T0 = to_opt4[0]
    C = to_opt4[1]
    n = to_opt4[2]
    Df = to_opt4[3]
    
    Rheo_guess = Df*(1-1/(C*(OBS_Time-T0)*(n-1)*Df**(n-1)+1)**(1/(n-1)))
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - Rheo_guess))
    Error_co_CB = np.dot(np.array(np.transpose(OBS_Data - Rheo_guess)),BC)
    
    denominator = np.dot(C_matrix_inv_selection,np.array(OBS_Data))
    ratio_denominator = np.dot(np.array(OBS_Data),denominator)
    ratio = Error_co_CB/ratio_denominator
    return ratio

def rheology_finder(creepmeter,event_dataframe,cov_matrix_inverse,creep_event_dataframes,event_start_time,event_durations,creep_data,which_event):
    duration_factor = 1
    #print(duration_factor*duration.iloc[j]/3)
    plt.close('all')
    Fitting_LV = {}
    Fitting_PLV = {}
    Fitting_VSF = {}
    Fitting_CB = {}
    idx=()
    for j in range(len(event_dataframe)):
        print(j)
        if len(creep_event_dataframes[which_event[j]].Time) <= len(cov_matrix_inverse):
            #if event_dataframe.rain_poss.iloc[j] == 0: #1 for rain, 0 for no rain
            if event_dataframe.rain_poss.iloc[j] == 1 or event_dataframe.rain_poss.iloc[j] == 0:
                time_to_plot = {}
                fitting_stepping_test_lv = {}
                misfit_stepping_test_lv = ()
                List_of_tests_lv = ()
                fitting_stepping_test_plv = {}
                misfit_stepping_test_plv = ()
                List_of_tests_plv = ()
                fitting_stepping_test_vsf = {}
                misfit_stepping_test_vsf = ()
                List_of_tests_vsf = ()
                fitting_stepping_test_cb = {}
                misfit_stepping_test_cb = ()
                List_of_tests_cb = ()
                plt.figure()
                plt.scatter(creep_event_dataframes[which_event[j]].Time,creep_event_dataframes[which_event[j]].Slip,label = 'Obs',s=5,c='black')
                for i in range((int(6*duration_factor*event_durations.iloc[j]/1.5))): #set how far throught the creep event to fit for... atm looks like it needs to be less than 1/3rd
                    boolarr = (event_start_time.iloc[j]+dt.timedelta(minutes=10*i) <= creep_data.Time) & (creep_data.Time <= event_start_time.iloc[j] + dt.timedelta(hours=duration_factor*event_durations.iloc[j]) - dt.timedelta(minutes=10))
                    Real_data2 = creep_data.Creep[boolarr]
                    t2 = creep_data.Time[boolarr]
                    V0_obs = creep_data.vel[boolarr]    
                    Real_data = Real_data2 - Real_data2.iloc[0]
                    t = (t2-t2.iloc[0])/dt.timedelta(hours=1)    
                    V_0_obs = V0_obs.iloc[0]
                    D_0_obs = Real_data.iloc[0]
                    D_f_obs = Real_data.iloc[-1]
                    C_matrix_inv_selection = cov_matrix_inverse[0:len(t),0:len(t)]
                    time_to_plot[i] = t
                    initial_guess_lv = (0,1,V_0_obs,D_0_obs)
                    initial_guess_plv = (0,1,V_0_obs,D_0_obs,1)
                    #initial_guess_vsf = (0,1,V_0_obs,D_0_obs,1)
                    initial_guess_vsf = (0,1,V_0_obs,D_0_obs)
                    initial_guess_cb = (0,1,1,D_f_obs)

                    res_lv = scipy.optimize.minimize(VIS_obj, initial_guess_lv,args=(t,Real_data,C_matrix_inv_selection), method = 'Nelder-Mead')
                    misfit_stepping_test_lv = np.append(misfit_stepping_test_lv,res_lv.fun)
                    List_of_tests_lv = np.append(List_of_tests_lv,i)
                    fitting_stepping_test_lv[i] = res_lv
                    misfit_stepping_test_lv = np.nan_to_num(misfit_stepping_test_lv,nan=999999)

                    res_plv = scipy.optimize.minimize(DUC_obj, initial_guess_plv, args=(t,Real_data,C_matrix_inv_selection), method = 'Nelder-Mead')
                    misfit_stepping_test_plv = np.append(misfit_stepping_test_plv,res_plv.fun)
                    List_of_tests_plv = np.append(List_of_tests_plv,i)
                    fitting_stepping_test_plv[i] = res_plv
                    misfit_stepping_test_plv = np.nan_to_num(misfit_stepping_test_plv,nan=999999)

                    res_vsf = scipy.optimize.minimize(VSF_obj, initial_guess_vsf, args=(t,Real_data,C_matrix_inv_selection) ,method = 'Nelder-Mead')
                    misfit_stepping_test_vsf = np.append(misfit_stepping_test_vsf,res_vsf.fun)
                    List_of_tests_vsf = np.append(List_of_tests_vsf,i)
                    fitting_stepping_test_vsf[i] = res_vsf
                    misfit_stepping_test_vsf = np.nan_to_num(misfit_stepping_test_vsf,nan=999999)

                    res_cb = scipy.optimize.minimize(CB_obj, initial_guess_cb, args=(t,Real_data,C_matrix_inv_selection) ,method = 'Nelder-Mead')
                    misfit_stepping_test_cb = np.append(misfit_stepping_test_cb,res_cb.fun)
                    List_of_tests_cb = np.append(List_of_tests_cb,i)
                    fitting_stepping_test_cb[i] = res_cb
                    misfit_stepping_test_cb = np.nan_to_num(misfit_stepping_test_cb,nan=999999)

                print('minimised')
                if len(misfit_stepping_test_lv) <1:
                    idx = np.append(idx,int(j))
                    continue
                else:
                    test_no_lv = np.array(np.where(misfit_stepping_test_lv == min(misfit_stepping_test_lv)))
                    res_lv_best = fitting_stepping_test_lv[test_no_lv[0][0]]
                    Plot_slip_shift_lv = creep_event_dataframes[which_event[j]].Slip.iloc[test_no_lv[0][0]] - creep_event_dataframes[which_event[j]].Slip.iloc[0] 
                    VIS_guess_plot = res_lv_best.x[3] + res_lv_best.x[2]*res_lv_best.x[1]*(1-np.exp(-(time_to_plot[test_no_lv[0][0]]-res_lv_best.x[0])/res_lv_best.x[1]))
                    plt.plot(time_to_plot[test_no_lv[0][0]]+(test_no_lv[0][0]*(1/6)),VIS_guess_plot + Plot_slip_shift_lv,color = 'purple',label ='Linear Viscous: T0:{q}, Misfit:{y}, Tau:{z}hrs'\
                         .format(q=creep_event_dataframes[which_event[j]].Time.iloc[test_no_lv[0][0]].round(3),y=res_lv_best.fun.round(3),z=res_lv_best.x[1].round(3)))

                if len(misfit_stepping_test_plv) <1:
                    idx = np.append(idx,int(j))
                    continue
                else:
                    test_no_plv = np.array(np.where(misfit_stepping_test_plv == min(misfit_stepping_test_plv)))
                    res_plv_best = fitting_stepping_test_plv[test_no_plv[0][0]]
                    Plot_slip_shift_plv = creep_event_dataframes[which_event[j]].Slip.iloc[test_no_plv[0][0]] - creep_event_dataframes[which_event[j]].Slip.iloc[0]
                    DUC_guess_plot = res_plv_best.x[3] + res_plv_best.x[2]*res_plv_best.x[1]*res_plv_best.x[4]*(1-(1+(1-1/res_plv_best.x[4])*((time_to_plot[test_no_plv[0][0]]-res_plv_best.x[0])/res_plv_best.x[1]))**(1/(1-res_plv_best.x[4])))
                    plt.plot(time_to_plot[test_no_plv[0][0]]+(test_no_plv[0][0]*1/6),DUC_guess_plot + Plot_slip_shift_plv,color = 'blue', label ='Power law viscous: T0:{q}, Misfit:{y}, Tau:{z}hrs'\
                             .format(q=creep_event_dataframes[which_event[j]].Time.iloc[test_no_plv[0][0]].round(3),y=res_plv_best.fun.round(3),z=res_plv_best.x[1].round(3)))

                if len(misfit_stepping_test_vsf) <1:
                    idx = np.append(idx,int(j))
                    continue
                else:
                    test_no_vsf = np.array(np.where(misfit_stepping_test_vsf == min(misfit_stepping_test_vsf)))
                    res_vsf_best = fitting_stepping_test_vsf[test_no_vsf[0][0]]
                    Plot_slip_shift_vsf = creep_event_dataframes[which_event[j]].Slip.iloc[test_no_vsf[0][0]] - creep_event_dataframes[which_event[j]].Slip.iloc[0]
                    #VSF_guess_plot = res_vsf_best.x[3] + res_vsf_best.x[4]*np.log(res_vsf_best.x[2]*(np.exp(1)**((t-res_vsf_best.x[0])/res_vsf_best.x[1])-1)+1)
                    VSF_guess_plot = res_vsf_best.x[3] + res_vsf_best.x[2]*res_vsf_best.x[1]*np.log(1+((time_to_plot[test_no_vsf[0][0]]-res_vsf_best.x[0])/res_vsf_best.x[1]))
                    #plt.plot(time_to_plot[test_no_vsf[0][0]]+(test_no_vsf[0][0]*1/6),VSF_guess_plot + Plot_slip_shift_vsf ,color = '#CC79A7', label ='Velocity Strengthening Friction: T0:{q}, Tau:{w}, V:{e}, D0:{r}, K:{u}, Misfit:{y}'\
                    #         .format(q=creep_event_dataframes[which_event[j]].Time.iloc[test_no_vsf[0][0]].round(3),w=res_vsf.x[1].round(3),e=res_vsf.x[2].round(3),r=res_vsf.x[3].round(3),u=res_vsf.x[4],y=res_vsf.fun.round(3)))
                    plt.plot(time_to_plot[test_no_vsf[0][0]]+(test_no_vsf[0][0]*1/6),VSF_guess_plot + Plot_slip_shift_vsf,color = 'red', label ='Velocity Strengthening Friction: T0:{q}, Misfit:{y}, Tau:{z}hrs'\
                             .format(q=creep_event_dataframes[which_event[j]].Time.iloc[test_no_vsf[0][0]].round(3),y=res_vsf_best.fun.round(3),z=res_vsf_best.x[1].round(3)))

                if len(misfit_stepping_test_cb)<1:
                    idx = np.append(idx,int(j))
                    continue
                else:            
                    test_no_cb = np.array(np.where(misfit_stepping_test_cb == min(misfit_stepping_test_cb)))
                    res_cb_best = fitting_stepping_test_cb[test_no_cb[0][0]]
                    Plot_slip_shift_cb = creep_event_dataframes[which_event[j]].Slip.iloc[test_no_cb[0][0]] - creep_event_dataframes[which_event[j]].Slip.iloc[0]    
                    CB_guess_plot = res_cb_best.x[3]*(1-1/(res_cb_best.x[1]*(time_to_plot[test_no_cb[0][0]]-res_cb_best.x[0])*(res_cb_best.x[2]-1)*res_cb_best.x[3]**(res_cb_best.x[2]-1)+1)**(1/(res_cb_best.x[2]-1)))
                    plt.plot(time_to_plot[test_no_cb[0][0]]+(test_no_cb[0][0]*1/6),CB_guess_plot + Plot_slip_shift_cb,color = 'orange',label ='Crough & Burford 1977: T0:{q} Misfit:{y}'\
                             .format(q=creep_event_dataframes[which_event[j]].Time.iloc[test_no_cb[0][0]].round(3),y=res_cb_best.fun.round(3)))

                plt.legend(fontsize=8)
                plt.xlabel('Hours after start of event',fontsize=14)
                plt.ylabel('Slip (mm)',fontsize=14)
                plt.title('{k}_{p}'.format(k=creepmeter,p=event_dataframe.og_index.iloc[j]),fontsize=14)
                #plt.show()
                
                Fitting_LV[which_event[j]] = res_lv_best
                Fitting_PLV[which_event[j]] = res_plv_best
                Fitting_VSF[which_event[j]] = res_vsf_best
                Fitting_CB[which_event[j]] = res_cb_best
            else:
                dummy=12
        
            figure = plt.gcf()  # get current figure
            figure.set_size_inches(8,8)
            plt.savefig("../../Rheology/{k}/Figures/{k}_{j}_event_plot_01_MAR_22.png".format(k=creepmeter,j=event_dataframe.og_index.iloc[j]),dpi=800)
            plt.close('all')
            dictionary_names = 'Fitting_LV','Fitting_PLV','Fitting_VSF','Fitting_CB'
            for i in range(len(dictionary_names)):
                with open("../../Rheology/{k}/{k}_{j}_fit_dictionary_01_MAR_22.txt".format(k=creepmeter,j=dictionary_names[i]), "wb") as tf:
                    pickle.dump(eval('{g}'.format(g=dictionary_names[i])),tf)
        else:
            idx=np.append(idx,int(j))
            continue
        with open("../../Rheology/{k}/{k}_events_skipped_01_MAR_22.txt".format(k=creepmeter),"wb") as tf2:
            pickle.dump(idx,tf2)
    #np.savetxt("../../Rheology/{k}/{k}_dropped_events.txt".format(k=creepmeter), idx)
    return Fitting_LV, Fitting_PLV, Fitting_VSF, Fitting_CB, idx


def pickle_load(CREEPMETER,dictionary_name):
    with open("../../Rheology/{k}/{k}_{j}_fit_dictionary_01_MAR_22.txt".format(k=CREEPMETER,j=dictionary_name), "rb") as tf:
        dictionary_load = pickle.load(tf)
    return dictionary_load


def misfit_extract(dictionary,which_event):
    COV_array = ()
    for i in range(len(dictionary)):
        COV_array = np.append(COV_array,dictionary[which_event[i]].fun)
    return COV_array