import numpy as np
import scipy
import matplotlib.pyplot as plt 
import datetime as dt
import pandas as pd
from scipy.optimize import minimize
import scipy


import Shape_fitter
import creep_part_identify as cpi
import seaborn as sns
import matplotlib.colors as Colors
from numba import jit
from multiprocessing import Pool
import os
import pickle


###################################################################################################
@jit(nopython=True,error_model = 'numpy')
def Linear_viscous(optimized_par,OBS_Time):
    T0 = optimized_par[0]
    Tau = optimized_par[1]
    V_0 = optimized_par[2] 
    Rheo_guess = V_0*Tau*(1-np.exp(-(OBS_Time-T0)/Tau))
    return Rheo_guess

@jit(nopython=True,error_model = 'numpy')
def Linear(optimized_par,OBS_Time):
    Vs = optimized_par[0]
    K = optimized_par[1]
    Fit_line = Vs*OBS_Time + K
    return Fit_line

@jit(nopython=True,error_model = 'numpy')
def Power_law_viscous(optimized_par,OBS_Time):
    T0 = optimized_par[0]
    Tau = optimized_par[1]
    V_0 = optimized_par[2]
    n = optimized_par[3]
    Rheo_guess  = V_0*Tau*n*(1-(1+(1-1/n)*((OBS_Time-T0)/Tau))**(1/(1-n)))
    return Rheo_guess

@jit(nopython=True,error_model = 'numpy')
def Velocity_strengthening_friction(optimized_par,OBS_Time):
    T0 = optimized_par[0]
    Tau = optimized_par[1]
    V_0 = optimized_par[2]   
    
    Rheo_guess = V_0*Tau*np.log(1+((OBS_Time-T0)/Tau))
    return Rheo_guess

@jit(nopython=True,error_model = 'numpy')
def Velocity_strengthening_friction_2(optimized_par,OBS_Time):
    T0 = optimized_par[0]
    Tau = optimized_par[1]
    V_0 = optimized_par[2]
    A_B = optimized_par[3]
    
    Rheo_guess = ((V_0*Tau)/(A_B-1))*(1-(1+(OBS_Time-T0)/Tau)**(1-A_B))
    return Rheo_guess

@jit(nopython=True,error_model = 'numpy')
def Velocity_strengthening_friction_3(optimized_par, OBS_Time):
    T0 = optimized_par[0]
    ta = optimized_par[1]
    V_0 = optimized_par[2]
    t1 = optimized_par[3]
    
    Rheo_guess = V_0*t1*np.log(1-(ta/t1)*(1-np.exp((OBS_Time - T0)/ta)))
    return Rheo_guess

@jit(nopython=True,error_model = 'numpy')
def Rate_dependent_friction(optimized_par, OBS_Time):
    T0 = optimized_par[0]
    ta = optimized_par[1]
    V_0 = optimized_par[2]
    V_0L = optimized_par[3]
    
    Rheo_guess = V_0*ta*np.log(1+(V_0L)*(np.exp((OBS_Time - T0)/ta)-1))
    return Rheo_guess

###################################################################################################
def LNV_dromedary(optimized_par,OBS_Time,OBS_Data,cov_matrix_inverse,no_phases,columns,j,CREEPMETER,file1):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Tau1 = optimized_par[5]
    V01 = optimized_par[6]
    T02 = optimized_par[7]
    S2 = optimized_par[8]
    
    C_matrix_inv_selection = cov_matrix_inverse[0:len(OBS_Time),0:len(OBS_Time)]
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Linear_viscous([T01,Tau1,V01],OBS_Time[jj])
    S2 = Vs*T01 + Linear_viscous([T01,Tau1,V01],T02)
    
    diff = np.array(OBS_Data - slip)    
    numerator = np.matmul(diff.T,np.matmul(C_matrix_inv_selection,diff))
    denominator = np.matmul(np.array(OBS_Data).T,np.matmul(C_matrix_inv_selection,np.array(OBS_Data)))
    ratio = numerator/denominator
    file1.write('{k} \n'.format(k=ratio))
    return ratio

def LNV_bactrian(optimized_par,OBS_Time,OBS_Data,cov_matrix_inverse,no_phases,columns,j,CREEPMETER):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Tau1 = optimized_par[5]
    V01 = optimized_par[6]
    T02 = optimized_par[7]
    S2 = optimized_par[8]
    Tau2 = optimized_par[9]
    V02 = optimized_par[10]
    T03 = optimized_par[11]
    S3 = optimized_par[12]
    
    C_matrix_inv_selection = cov_matrix_inverse[0:len(OBS_Time),0:len(OBS_Time)]
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Linear_viscous([T01,Tau1,V01],OBS_Time[jj])
    S2 = Vs*T01 + Linear_viscous([T01,Tau1,V01],T02)
    
    kk  = OBS_Time > T02
    slip[kk] = slip[kk] + Linear_viscous([T02,Tau2,V02],OBS_Time[kk])
    S3 = S2 + Linear_viscous([T02,Tau2,V02],T03)
    
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - slip))
    Error_co_VIS = np.dot(np.array(np.transpose(OBS_Data - slip)),BC)
    
    denominator = np.dot(C_matrix_inv_selection,np.array(OBS_Data))
    ratio_denominator = np.dot(np.array(OBS_Data),denominator)
    ratio = Error_co_VIS/ratio_denominator
    
    return ratio

def LNV_dromedary_plot(optimized_par,OBS_Time,no_phases):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Tau1 = optimized_par[5]
    V01 = optimized_par[6]
    T02 = optimized_par[7]
    S2 = optimized_par[8]
    
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Linear_viscous([T01,Tau1,V01],OBS_Time[jj])
    S2 = Vs*T01 + Linear_viscous([T01,Tau1,V01],T02)
    
    return slip

def LNV_bactrian_plot(optimized_par,OBS_Time,no_phases):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Tau1 = optimized_par[5]
    V01 = optimized_par[6]
    T02 = optimized_par[7]
    S2 = optimized_par[8]
    Tau2 = optimized_par[9]
    V02 = optimized_par[10]
    T03 = optimized_par[11]
    S3 = optimized_par[12]
    
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Linear_viscous([T01,Tau1,V01],OBS_Time[jj])
    S2 = Vs*T01 + Linear_viscous([T01,Tau1,V01],T02)
    
    kk  = OBS_Time > T02
    slip[kk] = slip[kk] + Linear_viscous([T02,Tau2,V02],OBS_Time[kk])
    S3 = S2 + Linear_viscous([T02,Tau2,V02],T03)
    
    return slip
###################################################################################################
def VSF_SS_dromedary(optimized_par,OBS_Time,OBS_Data,cov_matrix_inverse,no_phases,columns,j,CREEPMETER,file1):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Tau1 = optimized_par[5]
    V01 = optimized_par[6]
    T02 = optimized_par[7]
    S2 = optimized_par[8]
    
    C_matrix_inv_selection = cov_matrix_inverse[0:len(OBS_Time),0:len(OBS_Time)]
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Velocity_strengthening_friction([T01,Tau1,V01],OBS_Time[jj])
    S2 = Vs*T01 + Velocity_strengthening_friction([T01,Tau1,V01],T02)    
    
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - slip))
    Error_co_VIS = np.dot(np.array(np.transpose(OBS_Data - slip)),BC)
    
    diff = np.array(OBS_Data - slip)

    
    numerator = np.matmul(diff.T,np.matmul(C_matrix_inv_selection,diff))
    denominator = np.matmul(np.array(OBS_Data).T,np.matmul(C_matrix_inv_selection,np.array(OBS_Data)))

    ratio = numerator/denominator
    file1.write('{k} \n'.format(k=ratio))
    return ratio

def VSF_bactrian(optimized_par,OBS_Time,OBS_Data,cov_matrix_inverse,no_phases,columns,j,CREEPMETER):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Tau1 = optimized_par[5]
    V01 = optimized_par[6]
    T02 = optimized_par[7]
    S2 = optimized_par[8]
    Tau2 = optimized_par[9]
    V02 = optimized_par[10]
    T03 = optimized_par[11]
    S3 = optimized_par[12]
    
    C_matrix_inv_selection = cov_matrix_inverse[0:len(OBS_Time),0:len(OBS_Time)]
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Velocity_strengthening_friction([T01,Tau1,V01],OBS_Time[jj])
    S2 = Vs*T01 + Velocity_strengthening_friction([T01,Tau1,V01],T02)

    kk  = OBS_Time > T02
    slip[kk] = slip[kk] + Velocity_strengthening_friction([T02,Tau2,V02],OBS_Time[kk])
    S3 = S2 + Velocity_strengthening_friction([T02,Tau2,V02],T03)
    
    
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - slip))
    Error_co_VIS = np.dot(np.array(np.transpose(OBS_Data - slip)),BC)
    
    denominator = np.dot(C_matrix_inv_selection,np.array(OBS_Data))
    ratio_denominator = np.dot(np.array(OBS_Data),denominator)
    ratio = Error_co_VIS/ratio_denominator
    
    return ratio

def VSF_dromedary_plot(optimized_par,OBS_Time,no_phases):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Tau1 = optimized_par[5]
    V01 = optimized_par[6]
    T02 = optimized_par[7]
    S2 = optimized_par[8]
    
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Velocity_strengthening_friction([T01,Tau1,V01],OBS_Time[jj])
    S2 = Vs*T01 + Velocity_strengthening_friction([T01,Tau1,V01],T02)
    
    return slip

def VSF_bactrian_plot(optimized_par,OBS_Time,no_phases):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Tau1 = optimized_par[5]
    V01 = optimized_par[6]
    T02 = optimized_par[7]
    S2 = optimized_par[8]
    Tau2 = optimized_par[9]
    V02 = optimized_par[10]
    T03 = optimized_par[11]
    S3 = optimized_par[12]
    
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Velocity_strengthening_friction([T01,Tau1,V01],OBS_Time[jj])
    S2 = Vs*T01 + Velocity_strengthening_friction([T01,Tau1,V01],T02)

    kk  = OBS_Time > T02
    slip[kk] = slip[kk] + Velocity_strengthening_friction([T02,Tau2,V02],OBS_Time[kk])
    S3 = S2 + Velocity_strengthening_friction([T02,Tau2,V02],T03)
    
    return slip
###################################################################################################

def VSF_bSS_dromedary(optimized_par,OBS_Time,OBS_Data,cov_matrix_inverse,no_phases,columns,j,CREEPMETER,file1):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Tau1 = optimized_par[5]
    V01 = optimized_par[6]
    A_B1 = optimized_par[7]
    T02 = optimized_par[8]
    S2 = optimized_par[9]

    
    C_matrix_inv_selection = cov_matrix_inverse[0:len(OBS_Time),0:len(OBS_Time)]
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Velocity_strengthening_friction_2([T01,Tau1,V01,A_B1],OBS_Time[jj])
    S2 = Vs*T01 + Velocity_strengthening_friction_2([T01,Tau1,V01,A_B1],T02)

    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - slip))
    Error_co_VIS = np.dot(np.array(np.transpose(OBS_Data - slip)),BC)
    
    diff = np.array(OBS_Data - slip)

    
    numerator = np.matmul(diff.T,np.matmul(C_matrix_inv_selection,diff))
    denominator = np.matmul(np.array(OBS_Data).T,np.matmul(C_matrix_inv_selection,np.array(OBS_Data)))

    ratio = numerator/denominator
    file1.write('{k} \n'.format(k=ratio))
    return ratio

def VSF_bSS_dromedary_plot(optimized_par,OBS_Time,no_phases):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Tau1 = optimized_par[5]
    V01 = optimized_par[6]
    A_B1 = optimized_par[7]
    T02 = optimized_par[8]
    S2 = optimized_par[9]
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Velocity_strengthening_friction_2([T01,Tau1,V01,A_B1],OBS_Time[jj])
    S2 = Vs*T01 + Velocity_strengthening_friction_2([T01,Tau1,V01,A_B1],T02)

    return slip

###################################################################################################

def VSF_aSS_dromedary(optimized_par,OBS_Time,OBS_Data,cov_matrix_inverse,no_phases,columns,j,CREEPMETER,file1):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Ta1 = optimized_par[5]
    V01 = optimized_par[6]
    t1 = optimized_par[7]
    T02 = optimized_par[8]
    S2 = optimized_par[9]
    
    C_matrix_inv_selection = cov_matrix_inverse[0:len(OBS_Time),0:len(OBS_Time)]
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Velocity_strengthening_friction_3([T01,Ta1,V01,t1],OBS_Time[jj])
    S2 = Vs*T01 + Velocity_strengthening_friction_3([T01,Ta1,V01,t1],T02)
    
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - slip))
    Error_co_VIS = np.dot(np.array(np.transpose(OBS_Data - slip)),BC)
    
    denominator = np.dot(C_matrix_inv_selection,np.array(OBS_Data))
    ratio_denominator = np.dot(np.array(OBS_Data),denominator)
    ratio = Error_co_VIS/ratio_denominator
    
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - slip))
    Error_co_VIS = np.dot(np.array(np.transpose(OBS_Data - slip)),BC)
    
    diff = np.array(OBS_Data - slip)

    
    numerator = np.matmul(diff.T,np.matmul(C_matrix_inv_selection,diff))
    denominator = np.matmul(np.array(OBS_Data).T,np.matmul(C_matrix_inv_selection,np.array(OBS_Data)))

    ratio = numerator/denominator
    file1.write('{k} \n'.format(k=ratio))
    return ratio

def VSF3_bactrian(optimized_par,OBS_Time,OBS_Data,cov_matrix_inverse,no_phases,columns,j,CREEPMETER):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Ta1 = optimized_par[5]
    V01 = optimized_par[6]
    t1 = optimized_par[7]
    T02 = optimized_par[8]
    S2 = optimized_par[9]
    Ta2 = optimized_par[10]
    V02 = optimized_par[11]
    t2 = optimized_par[12]
    T03 = optimized_par[13]
    S3 = optimized_par[14]
    
    C_matrix_inv_selection = cov_matrix_inverse[0:len(OBS_Time),0:len(OBS_Time)]
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Velocity_strengthening_friction_3([T01,Ta1,V01,t1],OBS_Time[jj])
    S2 = Vs*T01 + Velocity_strengthening_friction_3([T01,Ta1,V01,t1],T02)

    kk  = OBS_Time > T02
    slip[kk] = slip[kk] + Velocity_strengthening_friction_3([T02,Ta2,V02,t2],OBS_Time[kk])
    S3 = S2 + Velocity_strengthening_friction_3([T02,Ta2,V02,t2],T03)
    
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - slip))
    Error_co_VIS = np.dot(np.array(np.transpose(OBS_Data - slip)),BC)
    
    denominator = np.dot(C_matrix_inv_selection,np.array(OBS_Data))
    ratio_denominator = np.dot(np.array(OBS_Data),denominator)
    ratio = Error_co_VIS/ratio_denominator
    
    return ratio

def VSF_aSS_dromedary_plot(optimized_par,OBS_Time,no_phases):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Ta1 = optimized_par[5]
    V01 = optimized_par[6]
    t1 = optimized_par[7]
    T02 = optimized_par[8]
    S2 = optimized_par[9]
    
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Velocity_strengthening_friction_3([T01,Ta1,V01,t1],OBS_Time[jj])
    S2 = Vs*T01 + Velocity_strengthening_friction_3([T01,Ta1,V01,t1],T02)


    return slip

def VSF3_bactrian_plot(optimized_par,OBS_Time,no_phases):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Ta1 = optimized_par[5]
    V01 = optimized_par[6]
    t1 = optimized_par[7]
    T02 = optimized_par[8]
    S2 = optimized_par[9]
    Ta2 = optimized_par[10]
    V02 = optimized_par[11]
    t2 = optimized_par[12]
    T03 = optimized_par[13]
    S3 = optimized_par[14]
    
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Velocity_strengthening_friction_3([T01,Ta1,V01,t1],OBS_Time[jj])
    S2 = Vs*T01 + Velocity_strengthening_friction_3([T01,Ta1,V01,t1],T02)

    kk  = OBS_Time > T02
    slip[kk] = slip[kk] + Velocity_strengthening_friction_3([T02,Ta2,V02,t2],OBS_Time[kk])
    S3 = S2 + Velocity_strengthening_friction_3([T02,Ta2,V02,t2],T03)

    return slip
###################################################################################################
def RDF_dromedary(optimized_par,OBS_Time,OBS_Data,cov_matrix_inverse,no_phases,columns,j,CREEPMETER,file1):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Ta1 = optimized_par[5]
    V01 = optimized_par[6]
    V_0L1 = optimized_par[7]
    T02 = optimized_par[8]
    S2 = optimized_par[9]

    
    C_matrix_inv_selection = cov_matrix_inverse[0:len(OBS_Time),0:len(OBS_Time)]
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Rate_dependent_friction([T01,Ta1,V01,V_0L1],OBS_Time[jj])
    S2 = Vs*T01 + Rate_dependent_friction([T01,Ta1,V01,V_0L1],T02)
    
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - slip))
    Error_co_VIS = np.dot(np.array(np.transpose(OBS_Data - slip)),BC)
    
    denominator = np.dot(C_matrix_inv_selection,np.array(OBS_Data))
    ratio_denominator = np.dot(np.array(OBS_Data),denominator)
    ratio = Error_co_VIS/ratio_denominator
    
    diff = np.array(OBS_Data - slip)

    numerator = np.matmul(diff.T,np.matmul(C_matrix_inv_selection,diff))
    denominator = np.matmul(np.array(OBS_Data).T,np.matmul(C_matrix_inv_selection,np.array(OBS_Data)))

    ratio = numerator/denominator
    file1.write('{k} \n'.format(k=ratio))
    return ratio

def RDF_bactrian(optimized_par,OBS_Time,OBS_Data,cov_matrix_inverse,no_phases,columns,j,CREEPMETER):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Ta1 = optimized_par[5]
    V01 = optimized_par[6]
    V_0L1 = optimized_par[7]
    T02 = optimized_par[8]
    S2 = optimized_par[9]
    Ta2 = optimized_par[10]
    V02 = optimized_par[11]
    V_0L2 = optimized_par[12]
    T03 = optimized_par[13]
    S3 = optimized_par[14]

    
    C_matrix_inv_selection = cov_matrix_inverse[0:len(OBS_Time),0:len(OBS_Time)]
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Rate_dependent_friction([T01,Ta1,V01,V_0L1],OBS_Time[jj])
    S2 = Vs*T01 + Rate_dependent_friction([T01,Ta1,V01,V_0L1],T02)

    kk  = OBS_Time > T02
    slip[kk] = slip[kk] + Rate_dependent_friction([T02,Ta2,V02,V_0L2],OBS_Time[kk])
    S3 = S2 + Rate_dependent_friction([T02,Ta2,V02,V_0L2],T03)
    
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - slip))
    Error_co_VIS = np.dot(np.array(np.transpose(OBS_Data - slip)),BC)
    
    denominator = np.dot(C_matrix_inv_selection,np.array(OBS_Data))
    ratio_denominator = np.dot(np.array(OBS_Data),denominator)
    ratio = Error_co_VIS/ratio_denominator
    
    return ratio

def RDF_dromedary_plot(optimized_par,OBS_Time,no_phases):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Ta1 = optimized_par[5]
    V01 = optimized_par[6]
    V_0L1 = optimized_par[7]
    T02 = optimized_par[8]
    S2 = optimized_par[9]
    
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Rate_dependent_friction([T01,Ta1,V01,V_0L1],OBS_Time[jj])
    S2 = Vs*T01 + Rate_dependent_friction([T01,Ta1,V01,V_0L1],T02)
    
    return slip

def RDF_bactrian_plot(optimized_par,OBS_Time,no_phases):
    Ts = optimized_par[0]
    Vs = optimized_par[1]
    K = optimized_par[2]
    T01 = optimized_par[3]
    S1 = optimized_par[4]
    Ta1 = optimized_par[5]
    V01 = optimized_par[6]
    V_0L1 = optimized_par[7]
    T02 = optimized_par[8]
    S2 = optimized_par[9]
    Ta2 = optimized_par[10]
    V02 = optimized_par[11]
    V_0L2 = optimized_par[12]
    T03 = optimized_par[13]
    S3 = optimized_par[14]

    
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Rate_dependent_friction([T01,Ta1,V01,V_0L1],OBS_Time[jj])
    S2 = Vs*T01 + Rate_dependent_friction([T01,Ta1,V01,V_0L1],T02)

    kk  = OBS_Time > T02
    slip[kk] = slip[kk] + Rate_dependent_friction([T02,Ta2,V02,V_0L2],OBS_Time[kk])
    S3 = S2 + Rate_dependent_friction([T02,Ta2,V02,V_0L2],T03)
    
    return slip
###################################################################################################
def PLV_dromedary(optimized_par,OBS_Time,OBS_Data,cov_matrix_inverse,no_phases,columns,j,CREEPMETER,file1):
    Ts =   optimized_par[0]
    Vs =   optimized_par[1]
    K =    optimized_par[2]
    T01 =  optimized_par[3]
    S1 =   optimized_par[4]
    Tau1 = optimized_par[5]
    V01 =  optimized_par[6]
    n1 =   optimized_par[7]
    T02 =  optimized_par[8]
    S2 =   optimized_par[9]
    
    C_matrix_inv_selection = cov_matrix_inverse[0:len(OBS_Time),0:len(OBS_Time)]
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Power_law_viscous([T01,Tau1,V01,n1],OBS_Time[jj])
    S2 = Vs*T01 + Power_law_viscous([T01,Tau1,V01,n1],T02)
    
    diff = np.array(OBS_Data - slip)

    
    numerator = np.matmul(diff.T,np.matmul(C_matrix_inv_selection,diff))
    denominator = np.matmul(np.array(OBS_Data).T,np.matmul(C_matrix_inv_selection,np.array(OBS_Data)))

    ratio = numerator/denominator
    file1.write('{k} \n'.format(k=ratio))    
    return ratio

def PLV_bactrian(optimized_par,OBS_Time,OBS_Data,cov_matrix_inverse,no_phases,columns,j,CREEPMETER):
    Ts =   optimized_par[0]
    Vs =   optimized_par[1]
    K =    optimized_par[2]
    T01 =  optimized_par[3]
    S1 =   optimized_par[4]
    Tau1 = optimized_par[5]
    V01 =  optimized_par[6]
    n1 =   optimized_par[7]
    T02 =  optimized_par[8]
    S2 =   optimized_par[9]
    Tau2 = optimized_par[10]
    V02 =  optimized_par[11]
    n2 =   optimized_par[12]   
    T03 =  optimized_par[13]
    S3 =   optimized_par[14]
    
    C_matrix_inv_selection = cov_matrix_inverse[0:len(OBS_Time),0:len(OBS_Time)]
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Power_law_viscous([T01,Tau1,V01,n1],OBS_Time[jj])
    S2 = Vs*T01 + Power_law_viscous([T01,Tau1,V01,n1],T02)

    kk  = OBS_Time > T02
    slip[kk] = slip[kk] + Power_law_viscous([T02,Tau2,V02,n2],OBS_Time[kk])
    S3 = S2 + Power_law_viscous([T02,Tau2,V02,n2],T03)
    
    
    BC = np.dot(C_matrix_inv_selection,np.array(OBS_Data - slip))
    Error_co_VIS = np.dot(np.array(np.transpose(OBS_Data - slip)),BC)
    
    denominator = np.dot(C_matrix_inv_selection,np.array(OBS_Data))
    ratio_denominator = np.dot(np.array(OBS_Data),denominator)
    ratio = Error_co_VIS/ratio_denominator
    
    return ratio

def PLV_dromedary_plot(optimized_par,OBS_Time,no_phases):
    Ts =   optimized_par[0]
    Vs =   optimized_par[1]
    K =    optimized_par[2]
    T01 =  optimized_par[3]
    S1 =   optimized_par[4]
    Tau1 = optimized_par[5]
    V01 =  optimized_par[6]
    n1 =   optimized_par[7]
    T02 =  optimized_par[8]
    S2 =   optimized_par[9]
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Power_law_viscous([T01,Tau1,V01,n1],OBS_Time[jj])
    S2 = Vs*T01 + Power_law_viscous([T01,Tau1,V01,n1],T02)
    return slip

def PLV_bactrian_plot(optimized_par,OBS_Time,no_phases):
    Ts =   optimized_par[0]
    Vs =   optimized_par[1]
    K =    optimized_par[2]
    T01 =  optimized_par[3]
    S1 =   optimized_par[4]
    Tau1 = optimized_par[5]
    V01 =  optimized_par[6]
    n1 =   optimized_par[7]
    T02 =  optimized_par[8]
    S2 =   optimized_par[9]
    Tau2 = optimized_par[10]
    V02 =  optimized_par[11]
    n2 =   optimized_par[12]   
    T03 =  optimized_par[13]
    S3 =   optimized_par[14]
    
    slip = np.zeros(len(OBS_Time))    
    ii = np.logical_and(OBS_Time >= Ts, OBS_Time <= T01)
    slip[ii] = Vs*OBS_Time[ii] + K
    S1 = Vs*T01 + K
    
    jj = OBS_Time > T01
    slip[jj] = Vs*T01 + Power_law_viscous([T01,Tau1,V01,n1],OBS_Time[jj])
    S2 = Vs*T01 + Power_law_viscous([T01,Tau1,V01,n1],T02)

    kk  = OBS_Time > T02
    slip[kk] = slip[kk] + Power_law_viscous([T02,Tau2,V02,n2],OBS_Time[kk])
    S3 = S2 + Power_law_viscous([T02,Tau2,V02,n2],T03)
    return slip

###################################################################################################

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
    creeping = pd.DataFrame({'Time':Time, 'Tm': Time,'Creep':min10_creep.astype(float)}) #create a pandas dataframe
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
    CWC_latlon = {'name': 'CWC', 'lat': 36.750 , 'lon': -121.385}
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
        if creepmeter == 'XHR' and j == 4:
            boolarr = (start.iloc[j] - dt.timedelta(hours=2)  <= creep_data.Time) & (creep_data.Time <= start.iloc[j] + dt.timedelta(hours=duration[j]) - dt.timedelta(hours=4)) #assumption made here
        else: 
            boolarr = (start.iloc[j] - dt.timedelta(hours=2) <= creep_data.Time) & (creep_data.Time <= start.iloc[j] + dt.timedelta(hours=duration[j])- dt.timedelta(minutes=10)) #assumption made here
        Creep_event_time = creep_data.Time[boolarr]
        Creep_event_time = (Creep_event_time - Creep_event_time.iloc[0])/dt.timedelta(hours=1) #set start time to 0
        Creep_event_slip = creep_data.Creep[boolarr]
        #Creep_event_slip = creep_data.rolling_mean[boolarr] #extract slip
        Creep_event_slip = Creep_event_slip - Creep_event_slip.iloc[0] #set initial slip to 0
        Velocity = creep_data.vel[boolarr]
        Acceleration = creep_data.acc[boolarr]
        dataframes[creep_index[j]] = pd.DataFrame({'Time':Creep_event_time,'Slip':Creep_event_slip,'Velocity':Velocity,'Acceleration':Acceleration})
        dataframes[creep_index[j]].reset_index(inplace=True)
    return dataframes, creep_index       

def creep_event_dataframe_short(dataframe,df_auto):
    dataframes={}
    creep_index = np.array(df_auto.og_index)
    for j in range(len(dataframe)):
        boolarr = dataframe[j].Slip <= 0.9*max(dataframe[j].Slip)
        SLIP = dataframe[j].Slip[boolarr]
        TIME = dataframe[j].Time[boolarr]
        VEL = dataframe[j].Velocity[boolarr]
        ACC = dataframe[j].Acceleration[boolarr]
        dataframes[creep_index[j]] = pd.DataFrame({'Time':TIME,'Slip':SLIP,'Velocity':VEL,'Acceleration':ACC})
        dataframes[creep_index[j]].reset_index(inplace=True)
    return dataframes, creep_index

###################################################################################################

def phase_splitter(Creep_Phase_no,dataframes):
        if len(Creep_Phase_no) == 4:
            P0 = np.logical_and(Creep_Phase_no.Ts <= np.array(dataframes.Time),  np.array(dataframes.Time) <  Creep_Phase_no.T01)
            P1 = (Creep_Phase_no.T01<= dataframes.Time)
            data_P0 = dataframes[P0]
            data_P1 = dataframes[P1]
            data_P1.reset_index(inplace=True)
            data_P2 = pd.DataFrame([[999999,data_P1.Slip.iloc[-1],0,0]],columns= ('Time','Slip','Velocity','Acceleration'))
            data_P3 = pd.DataFrame([[999999,data_P1.Slip.iloc[-1],0,0]],columns= ('Time','Slip','Velocity','Acceleration'))
            data_P4 = pd.DataFrame([[999999,data_P1.Slip.iloc[-1],0,0]],columns= ('Time','Slip','Velocity','Acceleration'))
            Creep_Phase_no['T02'] = data_P1.Time.iloc[-1]
            Creep_Phase_no['D02'] = data_P1.Slip.iloc[-1]
            Creep_Phase_no['T03'] = data_P1.Time.iloc[-1]
            Creep_Phase_no['D03'] = data_P1.Slip.iloc[-1]
            Creep_Phase_no['T04'] = data_P1.Time.iloc[-1]
            Creep_Phase_no['D04'] = data_P1.Slip.iloc[-1]
        elif len(Creep_Phase_no) == 6:
            P0 = np.logical_and(Creep_Phase_no.Ts <= np.array(dataframes.Time),  np.array(dataframes.Time) <  Creep_Phase_no.T01)
            P1 = np.logical_and(Creep_Phase_no.T01 <= np.array(dataframes.Time),  np.array(dataframes.Time) <  Creep_Phase_no.T02)
            P2 = (Creep_Phase_no.T02<= dataframes.Time)
            data_P0 = dataframes[P0]
            data_P1 = dataframes[P1]
            data_P1.reset_index(inplace=True)
            data_P2 = dataframes[P2]
            #print(data_P1)
            #print(data_P2)
            data_P2.reset_index(inplace=True)
            data_P3 = pd.DataFrame([[999999,data_P2.Slip.iloc[-1],0,0]],columns= ('Time','Slip','Velocity','Acceleration'))
            data_P4 = pd.DataFrame([[999999,data_P2.Slip.iloc[-1],0,0]],columns= ('Time','Slip','Velocity','Acceleration'))
            Creep_Phase_no['T03'] = data_P2.Time.iloc[-1]
            Creep_Phase_no['D03'] = data_P2.Slip.iloc[-1]
            Creep_Phase_no['T04'] = data_P2.Time.iloc[-1]
            Creep_Phase_no['D04'] = data_P2.Slip.iloc[-1]
        elif len(Creep_Phase_no) == 8:
            P0 = np.logical_and(Creep_Phase_no.Ts <= np.array(dataframes.Time),  np.array(dataframes.Time) <  Creep_Phase_no.T01)
            P1 = np.logical_and(Creep_Phase_no.T01 <= np.array(dataframes.Time),  np.array(dataframes.Time) <  Creep_Phase_no.T02)
            P2 = np.logical_and(Creep_Phase_no.T02 <= np.array(dataframes.Time),  np.array(dataframes.Time) <  Creep_Phase_no.T03)
            P3 = (Creep_Phase_no.T03<= dataframes.Time)
            data_P0 = dataframes[P0]
            data_P1 = dataframes[P1]
            data_P1.reset_index(inplace=True)
            data_P2 = dataframes[P2]
            data_P2.reset_index(inplace=True)
            data_P3 = dataframes[P3]
            data_P3.reset_index(inplace=True)
            data_P4 = pd.DataFrame([[999999,data_P3.Slip.iloc[-1],0,0]],columns= ('Time','Slip','Velocity','Acceleration'))
            Creep_Phase_no['T04'] = data_P3.Time.iloc[-1]
            Creep_Phase_no['D04'] = data_P3.Slip.iloc[-1]
        elif len(Creep_Phase_no) == 10:
            P0 = np.logical_and(Creep_Phase_no.Ts <= np.array(dataframes.Time),  np.array(dataframes.Time) <  Creep_Phase_no.T01)
            P1 = np.logical_and(Creep_Phase_no.T01 <= np.array(dataframes.Time),  np.array(dataframes.Time) <  Creep_Phase_no.T02)
            P2 = np.logical_and(Creep_Phase_no.T02 <= np.array(dataframes.Time),  np.array(dataframes.Time) <  Creep_Phase_no.T03)
            P3 = np.logical_and(Creep_Phase_no.T03 <= np.array(dataframes.Time),  np.array(dataframes.Time) <  Creep_Phase_no.T04)
            P4 = (Creep_Phase_no.T04<= dataframes.Time)
            data_P0 = dataframes[P0]
            data_P1 = dataframes[P1]
            data_P1.reset_index(inplace=True)
            data_P2 = dataframes[P2]
            data_P2.reset_index(inplace=True)
            data_P3 = dataframes[P3]
            data_P3.reset_index(inplace=True)
            data_P4 = dataframes[P4]
            data_P4.reset_index(inplace=True)
        
        return data_P0, data_P1, data_P2, data_P3, data_P4, Creep_Phase_no

def initial_and_bounds(creep_phase_new,data_P0,data_P1,data_P2,data_P3,data_P4,rheology):
    if rheology == 'LNV' or rheology == 'VSF_SS':
        initial_guess = [creep_phase_new.Ts,data_P0.iloc[0].Velocity,0,creep_phase_new.T01,creep_phase_new.D01,1,data_P1.iloc[0].Velocity,creep_phase_new.T02,creep_phase_new.D02,1,data_P2.iloc[0].Velocity,\
                        creep_phase_new.T03,creep_phase_new.D03,1,data_P3.iloc[0].Velocity,creep_phase_new.T04,creep_phase_new.D04,1,data_P4.iloc[0].Velocity]
        params = pd.DataFrame([initial_guess],columns = ('Ts','Vs','K','T01','S1','Tau1','V01','T02','S2','Tau2','V02','T03','S3','Tau3','V03','T04','S4','Tau4','V04'), index = ['initial'])                



        bnds = ((creep_phase_new.Ts-0.5, creep_phase_new.Ts+0.5),(0,max(data_P0.Velocity)),(-10,10), (creep_phase_new.T01-0.5, creep_phase_new.T01+0.5),(creep_phase_new.D01-0.01,creep_phase_new.D01+0.01),(0,100),(0,max(data_P1.Velocity)),\
                                                    (creep_phase_new.T02-0.5,creep_phase_new.T02+0.5),(creep_phase_new.D02-0.01,creep_phase_new.D02+0.01),(0,100),(0,max(data_P2.Velocity)),\
                                                    (creep_phase_new.T03-0.5,creep_phase_new.T03+0.5),(creep_phase_new.D03-0.01,creep_phase_new.D03+0.01),(0,100),(0,max(data_P3.Velocity)),\
                                                    (creep_phase_new.T04-0.5,creep_phase_new.T04+0.5),(creep_phase_new.D04-0.01,creep_phase_new.D04+0.01),(0,100),(0,max(data_P4.Velocity)))

        bnds_params = pd.DataFrame([bnds],columns = ('Ts','Vs','K','T01','S1','Tau1','V01','T02','S2','Tau2','V02','T03','S3','Tau3','V03','T04','S4','Tau4','V04'), index = ['bounds'])
        params = pd.concat([params,bnds_params])
    elif rheology == 'PLV':
        initial_guess = [creep_phase_new.Ts,data_P0.iloc[0].Velocity,0,creep_phase_new.T01,creep_phase_new.D01,1,data_P1.iloc[0].Velocity,1,creep_phase_new.T02,creep_phase_new.D02,1,data_P2.iloc[0].Velocity,1,\
                        creep_phase_new.T03,creep_phase_new.D03,1,data_P3.iloc[0].Velocity,1,creep_phase_new.T04,creep_phase_new.D04,1,data_P4.iloc[0].Velocity,1]
        params = pd.DataFrame([initial_guess],columns = ('Ts','Vs','K','T01','S1','Tau1','V01','n1','T02','S2','Tau2','V02','n2','T03','S3','Tau3','V03','n3','T04','S4','Tau4','V04','n4'), index = ['initial'])
        
        bnds = ((creep_phase_new.Ts-0.5, creep_phase_new.Ts+0.5),(0,max(data_P0.Velocity)),(-10,10),\
                (creep_phase_new.T01-0.5, creep_phase_new.T01+0.5),(creep_phase_new.D01-0.01,creep_phase_new.D01+0.01),(0,100),(0,max(data_P1.Velocity)),(0,5),\
                (creep_phase_new.T02-0.5,creep_phase_new.T02+0.5),(creep_phase_new.D02-0.01,creep_phase_new.D02+0.01),(0,100),(0,max(data_P2.Velocity)),(0,5),\
                (creep_phase_new.T03-0.5,creep_phase_new.T03+0.5),(creep_phase_new.D03-0.01,creep_phase_new.D03+0.01),(0,100),(0,max(data_P3.Velocity)),(0,5),\
                (creep_phase_new.T04-0.5,creep_phase_new.T04+0.5),(creep_phase_new.D04-0.01,creep_phase_new.D04+0.01),(0,100),(0,max(data_P4.Velocity)),(0,5))
        
        bnds_params = pd.DataFrame([bnds],columns = ('Ts','Vs','K','T01','S1','Tau1','V01','n1','T02','S2','Tau2','V02','n2','T03','S3','Tau3','V03','n3','T04','S4','Tau4','V04','n4'), index = ['bounds'])
        
        params = pd.concat([params,bnds_params])
        
    elif rheology == 'VSF_bSS':
        initial_guess = [creep_phase_new.Ts,data_P0.iloc[0].Velocity,0,creep_phase_new.T01,creep_phase_new.D01,1,data_P1.iloc[0].Velocity,1.01,creep_phase_new.T02,creep_phase_new.D02,1,data_P2.iloc[0].Velocity,1.01,\
                        creep_phase_new.T03,creep_phase_new.D03,1,data_P3.iloc[0].Velocity,1.01,creep_phase_new.T04,creep_phase_new.D04,1,data_P4.iloc[0].Velocity,1.01]
        params = pd.DataFrame([initial_guess],columns = ('Ts','Vs','K','T01','S1','Tau1','V01','A_B1','T02','S2','Tau2','V02','A_B2','T03','S3','Tau3','V03','A_B3','T04','S4','Tau4','V04','A_B4'), index = ['initial'])
        
        bnds = ((creep_phase_new.Ts-0.5, creep_phase_new.Ts+0.5),(0,max(data_P0.Velocity)),(-10,10),\
                (creep_phase_new.T01-0.5, creep_phase_new.T01+0.5),(creep_phase_new.D01-0.01,creep_phase_new.D01+0.01),(0,100),(0,max(data_P1.Velocity)),(0,10),\
                (creep_phase_new.T02-0.5,creep_phase_new.T02+0.5),(creep_phase_new.D02-0.01,creep_phase_new.D02+0.01),(0,100),(0,max(data_P2.Velocity)),(0,10),\
                (creep_phase_new.T03-0.5,creep_phase_new.T03+0.5),(creep_phase_new.D03-0.01,creep_phase_new.D03+0.01),(0,100),(0,max(data_P3.Velocity)),(0,10),\
                (creep_phase_new.T04-0.5,creep_phase_new.T04+0.5),(creep_phase_new.D04-0.01,creep_phase_new.D04+0.01),(0,100),(0,max(data_P4.Velocity)),(0,10))
        
        bnds_params = pd.DataFrame([bnds],columns = ('Ts','Vs','K','T01','S1','Tau1','V01','A_B1','T02','S2','Tau2','V02','A_B2','T03','S3','Tau3','V03','A_B3','T04','S4','Tau4','V04','A_B4'), index = ['bounds'])
        
        params = pd.concat([params,bnds_params])
        
    elif rheology == 'VSF_aSS':
        initial_guess = [creep_phase_new.Ts,data_P0.iloc[0].Velocity,0,creep_phase_new.T01,creep_phase_new.D01,1,data_P1.iloc[0].Velocity,1,creep_phase_new.T02,creep_phase_new.D02,1,data_P2.iloc[0].Velocity,1,\
                        creep_phase_new.T03,creep_phase_new.D03,1,data_P3.iloc[0].Velocity,1,creep_phase_new.T04,creep_phase_new.D04,1,data_P4.iloc[0].Velocity,1]
        params = pd.DataFrame([initial_guess],columns = ('Ts','Vs','K','T01','S1','Ta1','V01','t1','T02','S2','Ta2','V02','t2','T03','S3','Ta3','V03','t3','T04','S4','Ta4','V04','t4'), index = ['initial'])
        
        bnds = ((creep_phase_new.Ts-0.5, creep_phase_new.Ts+0.5),(0,max(data_P0.Velocity)),(-10,10),\
                (creep_phase_new.T01-0.5, creep_phase_new.T01+0.5),(creep_phase_new.D01-0.01,creep_phase_new.D01+0.01),(0,100),(0,max(data_P1.Velocity)),(0,10),\
                (creep_phase_new.T02-0.5,creep_phase_new.T02+0.5),(creep_phase_new.D02-0.01,creep_phase_new.D02+0.01),(0,100),(0,max(data_P2.Velocity)),(0,10),\
                (creep_phase_new.T03-0.5,creep_phase_new.T03+0.5),(creep_phase_new.D03-0.01,creep_phase_new.D03+0.01),(0,100),(0,max(data_P3.Velocity)),(0,10),\
                (creep_phase_new.T04-0.5,creep_phase_new.T04+0.5),(creep_phase_new.D04-0.01,creep_phase_new.D04+0.01),(0,100),(0,max(data_P4.Velocity)),(0,10))
        
        bnds_params = pd.DataFrame([bnds],columns = ('Ts','Vs','K','T01','S1','Ta1','V01','t1','T02','S2','Ta2','V02','t2','T03','S3','Ta3','V03','t3','T04','S4','Ta4','V04','t4'), index = ['bounds'])
        
        params = pd.concat([params,bnds_params])
        
    elif rheology == 'RDF':
        initial_guess = [creep_phase_new.Ts,data_P0.iloc[0].Velocity,0,creep_phase_new.T01,creep_phase_new.D01,1,data_P1.iloc[0].Velocity,1,creep_phase_new.T02,creep_phase_new.D02,1,data_P2.iloc[0].Velocity,1,\
                        creep_phase_new.T03,creep_phase_new.D03,1,data_P3.iloc[0].Velocity,1,creep_phase_new.T04,creep_phase_new.D04,1,data_P4.iloc[0].Velocity,1]
        params = pd.DataFrame([initial_guess],columns = ('Ts','Vs','K','T01','S1','Ta1','V01','V0/VL1','T02','S2','Ta2','V02','V0/VL2','T03','S3','Ta3','V03','V0/VL3','T04','S4','Ta4','V04','V0/VL4'), index = ['initial'])
        
        bnds = ((creep_phase_new.Ts-0.5, creep_phase_new.Ts+0.5),(0,max(data_P0.Velocity)),(-10,10),\
                (creep_phase_new.T01-0.5, creep_phase_new.T01+0.5),(creep_phase_new.D01-0.01,creep_phase_new.D01+0.01),(0,1000),(0,max(data_P1.Velocity)),(0,1000),\
                (creep_phase_new.T02-0.5,creep_phase_new.T02+0.5),(creep_phase_new.D02-0.01,creep_phase_new.D02+0.01),(0,1000),(0,max(data_P2.Velocity)),(0,1000),\
                (creep_phase_new.T03-0.5,creep_phase_new.T03+0.5),(creep_phase_new.D03-0.01,creep_phase_new.D03+0.01),(0,1000),(0,max(data_P3.Velocity)),(0,1000),\
                (creep_phase_new.T04-0.5,creep_phase_new.T04+0.5),(creep_phase_new.D04-0.01,creep_phase_new.D04+0.01),(0,1000),(0,max(data_P4.Velocity)),(0,1000))
        
        bnds_params = pd.DataFrame([bnds],columns = ('Ts','Vs','K','T01','S1','Ta1','V01','V0/VL1','T02','S2','Ta2','V02','V0/VL2','T03','S3','Ta3','V03','V0/VL3','T04','S4','Ta4','V04','V0/VL4'), index = ['bounds'])
        
        params = pd.concat([params,bnds_params])
        
    else:
        initial_guess = [creep_phase_new.Ts,data_P0.iloc[0].Velocity,0,creep_phase_new.T01,creep_phase_new.D01,data_P1.Slip.iloc[-1],1,1,creep_phase_new.T02,creep_phase_new.D02,data_P2.Slip.iloc[-1],1,1,\
                        creep_phase_new.T03,creep_phase_new.D03,data_P3.Slip.iloc[-1],1,1,creep_phase_new.T04,creep_phase_new.D04,data_P4.Slip.iloc[-1],1,1]
        params = pd.DataFrame([initial_guess],columns = ('Ts','Vs','K','T01','S1','Df1','C1','n1','T02','S2','Df2','C2','n2','T03','S3','Df3','C3','n3','T04','S4','Df4','C4','n4'), index = ['initial'])
        bnds = ((creep_phase_new.Ts-0.5, creep_phase_new.Ts+0.5),(0,max(data_P0.Velocity)),(-10,10),\
                (creep_phase_new.T01-0.5,creep_phase_new.T01+0.5),(creep_phase_new.D01-0.01,creep_phase_new.D01+0.01),(data_P1.Slip.iloc[-1]-0.01,data_P1.Slip.iloc[-1]+0.01),(0,5E-3),(0,5),\
                (creep_phase_new.T02-0.5,creep_phase_new.T02+0.5),(creep_phase_new.D02-0.01,creep_phase_new.D02+0.01),(data_P2.Slip.iloc[-1]-0.01,data_P2.Slip.iloc[-1]+0.01),(0,5E-3),(0,5),\
                (creep_phase_new.T03-0.5,creep_phase_new.T03+0.5),(creep_phase_new.D03-0.01,creep_phase_new.D03+0.01),(data_P3.Slip.iloc[-1]-0.01,data_P3.Slip.iloc[-1]+0.01),(0,5E-3),(0,5),\
                (creep_phase_new.T04-0.5,creep_phase_new.T04+0.5),(creep_phase_new.D04-0.01,creep_phase_new.D04+0.01),(data_P4.Slip.iloc[-1]-0.01,data_P4.Slip.iloc[-1]+0.01),(0,5E-3),(0,5))
        bnds_params = pd.DataFrame([bnds],columns = ('Ts','Vs','K','T01','S1','Df1','C1','n1','T02','S2','Df2','C2','n2','T03','S3','Df3','C3','n3','T04','S4','Df4','C4','n4'), index = ['bounds'])
        
        params = pd.concat([params,bnds_params])
    return params

def check_dir(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(path, exist_ok=True)
###################################################################################################
def LNV_cf_dromedary(t,Ts,Vs,K,T01,S1,Tau1,V01,T02,S2):
       
    slip = np.zeros(len(t))    
    ii = np.logical_and(t >= Ts, t <= T01)
    slip[ii] = Vs*t[ii] + K
    S1 = Vs*T01 + K
    
    jj = t > T01
    slip[jj] = Vs*T01 + Linear_viscous([T01,Tau1,V01],t[jj])
    S2 = Vs*T01 + Linear_viscous([T01,Tau1,V01],T02)
    
    return slip

def RDF_cf_dromedary(t,Ts,Vs,K,T01,S1,Ta1,V01,V_0L1,T02,S2):
    
    slip = np.zeros(len(t))    
    ii = np.logical_and(t >= Ts, t <= T01)
    slip[ii] = Vs*t[ii] + K
    S1 = Vs*T01 + K
    
    jj = t > T01
    slip[jj] = Vs*T01 + Rate_dependent_friction([T01,Ta1,V01,V_0L1],t[jj])
    S2 = Vs*T01 + Rate_dependent_friction([T01,Ta1,V01,V_0L1],T02)
    
    return slip   
 ###################################################################################################
def LNV_fitter(time,slip,cov_matrix_inv,no_phases,columns_LNV,j,CREEPMETER,LNV_DF_params,atest,file_misfit):
    print('Linear Viscous: {k}'.format(k=j))
    print ("Current date and time : ")
    
    print (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    isExistLNV = os.path.exists('/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{y}/LNV/{k}_{y}_LNV_fit_dictionary_multi_phase_27_APR_23.txt'.format(k=CREEPMETER,y=j))
    if not isExistLNV:
        success = False
        n_iter = 5000
        LNV_bounds = LNV_DF_params.loc['bounds'].to_list()
        LNV_initial_guess = LNV_DF_params.loc['initial'].to_list()
        while success == False:
            if success == False:
                res_LNV = scipy.optimize.basinhopping(LNV_dromedary, LNV_initial_guess,\
                accept_test = atest, minimizer_kwargs = ({'args':(time,slip,cov_matrix_inv,no_phases,\
                columns_LNV,j,CREEPMETER,file_misfit),'method':'SLSQP','bounds':LNV_bounds}))#,'options':{'maxiter':n_iter}}),niter=1000)            
                n_iter = n_iter+2000
                success = res_LNV.success
                LNV_initial_guess = res_LNV.x
        dictionary_LNV = {}
        dictionary_LNV['fit'] = res_LNV
        LNV_fitting_params = pd.DataFrame([res_LNV.x],columns = ('Ts','Vs','K','T01','S1','Tau1','V01','T02','S2'), index = ['fitted'])
        LNV_DF_params = pd.concat([LNV_DF_params,LNV_fitting_params])
        dictionary_LNV['fitting params'] = LNV_DF_params
        #dictionary_LNV['params tried'] = LNV_params_tried
        with open("/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{u}/LNV/{k}_{u}_LNV_fit_dictionary_multi_phase_27_APR_23.txt".format(k=CREEPMETER,u=j), "wb") as tf:
            pickle.dump(dictionary_LNV,tf)
    else:
        with open("/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{u}/LNV/{k}_{u}_LNV_fit_dictionary_multi_phase_27_APR_23.txt".format(k=CREEPMETER,u=j), "rb") as tf:
            dictionary_LNV = pickle.load(tf)
            res_LNV = dictionary_LNV['fit']
            LNV_DF_params = dictionary_LNV['fitting params']
    return LNV_DF_params

###################################################################################################
def PLV_fitter(time,slip,cov_matrix_inv,no_phases,columns_PLV,j,CREEPMETER,PLV_DF_params,atest,file_misfit):
    print('Power-law Viscous: {k}'.format(k=j))
    print ("Current date and time : ")
    print (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    isExistPLV = os.path.exists('/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{y}/PLV/{k}_{y}_PLV_fit_dictionary_multi_phase_27_APR_23.txt'.format(k=CREEPMETER,y=j))
    if not isExistPLV:
        #print('../../Rheology/Single_rheology_04_APR_23/{k}/fits/{y}/PLV/{k}_{y}_PLV_fit_dictionary_multi_phase_04_APR_23.txt'.format(k=CREEPMETER[q],y=j))
        success = False
        n_iter = 5000
        PLV_bounds = PLV_DF_params.loc['bounds'].to_list()
        PLV_initial_guess = PLV_DF_params.loc['initial'].to_list()
        while success == False:
            if success == False:
                res_PLV = scipy.optimize.basinhopping(PLV_dromedary, PLV_initial_guess,\
                accept_test= atest, minimizer_kwargs = ({'args':(time,slip,cov_matrix_inv,no_phases,\
                columns_PLV,j,CREEPMETER,file_misfit),'method':'SLSQP','bounds':PLV_bounds}))#,'options':{'maxiter':n_iter}}),niter=1000)
                n_iter = n_iter+2000
                success = res_PLV.success
                PLV_initial_guess = res_PLV.x

        dictionary_PLV = {}
        dictionary_PLV['fit'] = res_PLV
        PLV_fitting_params = pd.DataFrame([res_PLV.x],columns = ('Ts','Vs','K','T01','S1','Tau1','V01','n1','T02','S2'), index = ['fitted'])
        PLV_DF_params = pd.concat([PLV_DF_params,PLV_fitting_params])
        dictionary_PLV['fitting params'] = PLV_DF_params
        #dictionary_PLV['params tried'] = PLV_params_tried
        with open("/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{u}/PLV/{k}_{u}_PLV_fit_dictionary_multi_phase_27_APR_23.txt".format(k=CREEPMETER,u=j), "wb") as tf:
            pickle.dump(dictionary_PLV,tf)
    else:
        with open("/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{u}/PLV/{k}_{u}_PLV_fit_dictionary_multi_phase_27_APR_23.txt".format(k=CREEPMETER,u=j), "rb") as tf:
            dictionary_PLV = pickle.load(tf)
            res_PLV = dictionary_PLV['fit']
            PLV_DF_params = dictionary_PLV['fitting params']
    return PLV_DF_params
###################################################################################################
def VSF_SS_fitter(time,slip,cov_matrix_inv,no_phases,columns_VSF_SS,j,CREEPMETER,VSF_SS_DF_params,atest,file_misfit):
    print('VSF-SS: {k}'.format(k=j))
    print ("Current date and time : ")
    print (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    isExistVSF_SS = os.path.exists('/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{y}/VSF_SS/{k}_{y}_VSF_SS_fit_dictionary_multi_phase_27_APR_23.txt'.format(k=CREEPMETER,y=j))
    if not isExistVSF_SS:
        #print('../../Rheology/Single_rheology_27_APR_23/{k}/fits/{y}/VSF_SS/{k}_{y}_VSF_SS_fit_dictionary_multi_phase_27_APR_23.txt'.format(k=CREEPMETER[q],y=j))
        success = False
        n_iter = 5000
        VSF_SS_bounds = VSF_SS_DF_params.loc['bounds'].to_list()
        VSF_SS_initial_guess = VSF_SS_DF_params.loc['initial'].to_list()
        while success == False:
            if success == False:
                res_VSF_SS = scipy.optimize.basinhopping(VSF_SS_dromedary, VSF_SS_initial_guess,\
                accept_test = atest, minimizer_kwargs = ({'args':(time,slip,cov_matrix_inv,no_phases,\
                columns_VSF_SS,j,CREEPMETER,file_misfit),'method':'SLSQP','bounds':VSF_SS_bounds}))#,'options':{'maxiter':n_iter}}),niter=1000)
                n_iter = n_iter+2000
                success = res_VSF_SS.success
                VSF_SS_initial_guess = res_VSF_SS.x

        dictionary_VSF_SS = {}
        dictionary_VSF_SS['fit'] = res_VSF_SS
        VSF_SS_fitting_params = pd.DataFrame([res_VSF_SS.x],columns = ('Ts','Vs','K','T01','S1','Tau1','V01','T02','S2'), index = ['fitted'])
        VSF_SS_DF_params = pd.concat([VSF_SS_DF_params,VSF_SS_fitting_params])
        dictionary_VSF_SS['fitting params'] = VSF_SS_DF_params
        #dictionary_VSF_SS['params tried'] = VSF_SS_params_tried
        with open("/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{u}/VSF_SS/{k}_{u}_VSF_SS_fit_dictionary_multi_phase_27_APR_23.txt".format(k=CREEPMETER,u=j), "wb") as tf:
            pickle.dump(dictionary_VSF_SS,tf)
    else:
        with open("/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{u}/VSF_SS/{k}_{u}_VSF_SS_fit_dictionary_multi_phase_27_APR_23.txt".format(k=CREEPMETER,u=j), "rb") as tf:
            dictionary_VSF_SS = pickle.load(tf)
            res_VSF_SS = dictionary_VSF_SS['fit']
            VSF_SS_DF_params = dictionary_VSF_SS['fitting params']
    return VSF_SS_DF_params
###################################################################################################
def RDF_fitter(time,slip,cov_matrix_inv,no_phases,columns_RDF,j,CREEPMETER,RDF_DF_params,atest,file_misfit):
    print('RDF: {k}'.format(k=j))
    print ("Current date and time : ")
    print (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    isExistRDF = os.path.exists('../../Rheology/Single_rheology_27_APR_23/{k}/fits/{y}/RDF/{k}_{y}_RDF_fit_dictionary_multi_phase_27_APR_23.txt'.format(k=CREEPMETER,y=j))
    if not isExistRDF:
        #print('../../Rheology/Single_rheology_27_APR_23/{k}/fits/{y}/RDF/{k}_{y}_RDF_fit_dictionary_multi_phase_27_APR_23.txt'.format(k=CREEPMETER[q],y=j))
        success = False
        n_iter = 5000
        RDF_bounds = RDF_DF_params.loc['bounds'].to_list()
        RDF_initial_guess = RDF_DF_params.loc['initial'].to_list()
        while success == False:
            if success == False:
                res_RDF = scipy.optimize.basinhopping(RDF_dromedary, RDF_initial_guess,\
                accept_test = atest, minimizer_kwargs = ({'args':(time,slip,cov_matrix_inv,no_phases,\
                columns_RDF,j,CREEPMETER,file_misfit),'method':'SLSQP','bounds':RDF_bounds}))#,'options':{'maxiter':n_iter}}),niter=1000)
                n_iter = n_iter+2000
                success = res_RDF.success
                RDF_initial_guess = res_RDF.x
        dictionary_RDF = {}
        dictionary_RDF['fit'] = res_RDF
        RDF_fitting_params = pd.DataFrame([res_RDF.x],columns = ('Ts','Vs','K','T01','S1','Ta1','V01','V0/VL1','T02','S2'), index = ['fitted'])
        RDF_DF_params = pd.concat([RDF_DF_params,RDF_fitting_params])
        dictionary_RDF['fitting params'] = RDF_DF_params
        #dictionary_RDF['params tried'] = RDF_params_tried
        with open("/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{u}/RDF/{k}_{u}_RDF_fit_dictionary_multi_phase_27_APR_23.txt".format(k=CREEPMETER,u=j), "wb") as tf:
            pickle.dump(dictionary_RDF,tf)

    else:
        with open("/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{u}/RDF/{k}_{u}_RDF_fit_dictionary_multi_phase_27_APR_23.txt".format(k=CREEPMETER,u=j), "rb") as tf:
            dictionary_RDF = pickle.load(tf)
            res_RDF = dictionary_RDF['fit']
            RDF_DF_params = dictionary_RDF['fitting params']
    return RDF_DF_params
###################################################################################################
def VSF_aSS_fitter(time,slip,cov_matrix_inv,no_phases,columns_VSF_aSS,j,CREEPMETER,VSF_aSS_DF_params,atest,file_misfit):
    print('VSF_aSS: {k}'.format(k=j))
    print ("Current date and time : ")
    print (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    isExistVSF_aSS = os.path.exists('../../Rheology/Single_rheology_27_APR_23/{k}/fits/{y}/VSF_aSS/{k}_{y}_VSF_aSS_fit_dictionary_multi_phase_27_APR_23.txt'.format(k=CREEPMETER,y=j))
    if not isExistVSF_aSS:
        #print('../../Rheology/Single_rheology_27_APR_23/{k}/fits/{y}/VSF_aSS/{k}_{y}_VSF_aSS_fit_dictionary_multi_phase_27_APR_23.txt'.format(k=CREEPMETER[q],y=j))
        success = False
        n_iter = 5000
        VSF_aSS_bounds = VSF_aSS_DF_params.loc['bounds'].to_list()
        VSF_aSS_initial_guess = VSF_aSS_DF_params.loc['initial'].to_list()
        while success == False:
            if success == False:
                res_VSF_aSS = scipy.optimize.basinhopping(VSF_aSS_dromedary, VSF_aSS_initial_guess,\
                accept_test = atest, minimizer_kwargs = ({'args':(time,slip,cov_matrix_inv,no_phases,\
                columns_VSF_aSS,j,CREEPMETER,file_misfit),'method':'SLSQP','bounds':VSF_aSS_bounds}))#,'options':{'maxiter':n_iter}}),niter=1000)
                n_iter = n_iter+2000
                success = res_VSF_aSS.success
                VSF_aSS_initial_guess = res_VSF_aSS.x
        dictionary_VSF_aSS = {}
        dictionary_VSF_aSS['fit'] = res_VSF_aSS
        VSF_aSS_fitting_params = pd.DataFrame([res_VSF_aSS.x],columns = ('Ts','Vs','K','T01','S1','Ta1','V01','t1','T02','S2'), index = ['fitted'])
        VSF_aSS_DF_params = pd.concat([VSF_aSS_DF_params,VSF_aSS_fitting_params])
        dictionary_VSF_aSS['fitting params'] = VSF_aSS_DF_params
        #dictionary_VSF_aSS['params tried'] = VSF_aSS_params_tried
        with open("/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{u}/VSF_aSS/{k}_{u}_VSF_aSS_fit_dictionary_multi_phase_27_APR_23.txt".format(k=CREEPMETER,u=j), "wb") as tf:
            pickle.dump(dictionary_VSF_aSS,tf)

    else:
        with open("/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{u}/VSF_aSS/{k}_{u}_VSF_aSS_fit_dictionary_multi_phase_27_APR_23.txt".format(k=CREEPMETER,u=j), "rb") as tf:
            dictionary_VSF_aSS = pickle.load(tf)
            res_VSF_aSS = dictionary_VSF_aSS['fit']
            VSF_aSS_DF_params = dictionary_VSF_aSS['fitting params']
    return VSF_aSS_DF_params
###################################################################################################
def VSF_bSS_fitter(time,slip,cov_matrix_inv,no_phases,columns_VSF_bSS,j,CREEPMETER,VSF_bSS_DF_params,atest,file_misfit):
    print('VSF-bSS: {k}'.format(k=j))
    print ("Current date and time : ")
    print (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    isExistVSF_bSS = os.path.exists('/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{y}/VSF_bSS/{k}_{y}_VSF_bSS_fit_dictionary_multi_phase_27_APR_23.txt'.format(k=CREEPMETER,y=j))
    if not isExistVSF_bSS:
        #print('../../Rheology/Single_rheology_27_APR_23/{k}/fits/{y}/VSF_SS/{k}_{y}_VSF_bSS_fit_dictionary_multi_phase_27_APR_23.txt'.format(k=CREEPMETER[q],y=j))
        success = False
        n_iter = 5000
        VSF_bSS_bounds = VSF_bSS_DF_params.loc['bounds'].to_list()
        #print(VSF_bSS_bounds)
        VSF_bSS_initial_guess = VSF_bSS_DF_params.loc['initial'].to_list()
        while success == False:
            if success == False:
                res_VSF_bSS = scipy.optimize.basinhopping(VSF_bSS_dromedary, VSF_bSS_initial_guess,\
                accept_test = atest, minimizer_kwargs = ({'args':(time,slip,cov_matrix_inv,no_phases,\
                columns_VSF_bSS,j,CREEPMETER,file_misfit),'method':'SLSQP','bounds':VSF_bSS_bounds}))#,'options':{'maxiter':n_iter}}),niter=1000)
                n_iter = n_iter+2000
                success = res_VSF_bSS.success
                VSF_bSS_initial_guess = res_VSF_bSS.x

        dictionary_VSF_bSS = {}
        dictionary_VSF_bSS['fit'] = res_VSF_bSS
        VSF_bSS_fitting_params = pd.DataFrame([res_VSF_bSS.x],columns = ('Ts','Vs','K','T01','S1','Tau1','V01','A_B1','T02','S2'), index = ['fitted'])
        VSF_bSS_DF_params = pd.concat([VSF_bSS_DF_params,VSF_bSS_fitting_params])
        dictionary_VSF_bSS['fitting params'] = VSF_bSS_DF_params
        #dictionary_VSF_SS['params tried'] = VSF_bSS_params_tried
        with open("/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{u}/VSF_bSS/{k}_{u}_VSF_bSS_fit_dictionary_multi_phase_27_APR_23.txt".format(k=CREEPMETER,u=j), "wb") as tf:
            pickle.dump(dictionary_VSF_bSS,tf)
    else:
        with open("/home/earthquakes2/homes/Dan/Rheology/Single_rheology_27_APR_23/{k}/fits/{u}/VSF_bSS/{k}_{u}_VSF_bSS_fit_dictionary_multi_phase_27_APR_23.txt".format(k=CREEPMETER,u=j), "rb") as tf:
            dictionary_VSF_bSS = pickle.load(tf)
            res_VSF_SS = dictionary_VSF_bSS['fit']
            VSF_bSS_DF_params = dictionary_VSF_bSS['fitting params']
    return VSF_bSS_DF_params
###################################################################################################