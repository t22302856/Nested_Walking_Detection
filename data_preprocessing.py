#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 4 2024

@author: kai-chunliu
"""

import numpy as np
import pandas as pd
from math import ceil
from utilities import signalPlot
from collections import Counter

from sklearn.preprocessing import StandardScaler

def normalize(config, c_data):
    if config['Normalization'] == 'z-score':
        scaler = StandardScaler(with_std=False)
        scaler.fit(c_data)
        c_data_normalize = scaler.fit_transform(c_data)
        return c_data_normalize
    else:
        return c_data
    scaler = StandardScaler()
    scaler.fit(c_data)
    c_data_normalize = scaler.fit_transform(c_data)
    print(scaler.transform(c_data))

# from DAManager import DAManager

def SlidindWindow_and_lables(config,c_data, c_data_label):

    
    # initialize first window
    j=0
    Window_list=np.zeros((1,config['window_size'],config['nb_channels']),dtype=np.float32)
    Window_list[0,:,:]=c_data.iloc[j:config['window_size'],:]
    
    j=int(config['overlap'])
    while (j+config['window_size']) <= len(c_data):
        Window=np.zeros((1,config['window_size'],config['nb_channels']),dtype=np.float32)
        Window[0,:,:]=c_data.iloc[j:j+config['window_size'],:]
        Window_list = np.concatenate((Window_list,Window))
        j=j+config['overlap']
    
    #assign labels
    label1_list = np.zeros((np.size(Window_list,0)),dtype=np.int64)
    label2_list = np.zeros((np.size(Window_list,0)),dtype=np.int64)
    label1_list[:] =  c_data_label['num_Label1']-1   
    label2_list[:] =  c_data_label['num_Label2']
 
    
    return Window_list, label1_list, label2_list

def SlidindWindow_and_lables_ADL(config,c_data, c_data_label):

    
    # initialize first window
    j=0
    Window_list=np.zeros((1,config['window_size'],config['nb_channels']),dtype=np.float32)
    Window_list[0,:,:]=c_data.iloc[j:config['window_size'],:]
    Label_list = np.zeros((np.size(Window_list,0)),dtype=np.int64)
    Label = np.zeros((np.size(Label_list,0)),dtype=np.int64)
    j=int(config['overlap'])
    while (j+config['window_size']) <= len(c_data):
        
        # assign label 
        c_Window_label = c_data_label[j:j+config['window_size']]
        walking_count = c_Window_label[c_Window_label==1].count()
        if walking_count>150:
            Label[0] = 1
            Label_list = np.concatenate((Label_list,Label))
            Discard = 0
        elif -1 in c_data_label[j:j+config['window_size']].values :
            Discard = 1
        else:
            Label[0] = 0
            Label_list = np.concatenate((Label_list,Label))
            Discard = 0
        
        if Discard == 0:
            # sliding window
            Window=np.zeros((1,config['window_size'],config['nb_channels']),dtype=np.float32)
            Window[0,:,:]=c_data.iloc[j:j+config['window_size'],:]
            Window_list = np.concatenate((Window_list,Window))
        
        
        
        j=j+config['overlap']    
    
    return Window_list, Label_list

    
    
def create_segments_and_labels(config,df):
    DataList=df['Folder'].values[:]
    
    #initialize with zero
    Window_list = np.zeros((1,config['window_size'],config['nb_channels']),dtype=np.float32)
    label1_list = np.zeros((np.size(Window_list,0)),dtype=np.int64)
    label2_list = np.zeros((np.size(Window_list,0)),dtype=np.int64)
    subjectID_list = np.zeros((np.size(Window_list,0)),dtype=np.int64)
    
    #read complete data
    A = []
    for i in range(len(DataList)):
        data = pd.read_csv(config['load_path']+df['Folder'].values[i]+'/ag.csv',header=None)
        label_table =  pd.read_csv(config['load_path']+df['Folder'].values[i]+'/LabelInfo2.csv')
        
        #read each activity sequence
        for j in range(np.shape(label_table)[0]): 
            Start = label_table['Starting'][j]
            End = label_table['Ending'][j]
            c_data_label = label_table.iloc[j,:]
            
            # zero padding while the labeled duration is less than window size
            if (End-Start)<(config['window_size']+config['overlap']*config['fold']):
                c_data1 = data.loc[Start:End,:]  
                extendSize = ceil((config['window_size']+config['overlap']*config['fold']-(End-Start))/2)     
                eStart = Start-extendSize
                eEnd = End+extendSize
                c_data = data.loc[eStart:eEnd,:].copy()  
                c_data.loc[eStart:Start, :] = 0
                c_data.loc[End:eEnd] =0
            else:
                c_data = data.loc[Start:End,:]  
            
            # sliding window and assign labels
            c_Window_list, c_label1_list, c_label2_list= SlidindWindow_and_lables(config,c_data, c_data_label)    
            c_subjectID_list = np.zeros((np.size(c_Window_list,0)),dtype=np.int64)
            c_subjectID_list[:] =  df['subjID'].iloc[i]  
            
            Window_list = np.concatenate((Window_list,c_Window_list))
            label1_list = np.concatenate((label1_list,c_label1_list))
            label2_list = np.concatenate((label2_list,c_label2_list))
            subjectID_list = np.concatenate((subjectID_list,c_subjectID_list))
            
            
        
    
    # delete the 1st row
    Window_list = np.delete(Window_list, 0, axis=0)
    label1_list = np.delete(label1_list, 0, axis=0)
    label2_list = np.delete(label2_list, 0, axis=0)
    subjectID_list = np.delete(subjectID_list, 0, axis=0)
    
    return Window_list, label1_list, label2_list, subjectID_list

def create_segments_and_labels_ADL(config,df):
    DataList=df['Folder'].values[:]
    
    #initialize with zero
    Window_list = np.zeros((1,config['window_size'],config['nb_channels']),dtype=np.float32)
    label1_list = np.zeros((np.size(Window_list,0)),dtype=np.int64)
    subjectID_list = np.zeros((np.size(Window_list,0)),dtype=np.int64)
    

    for i in range(len(DataList)):
        data_r = pd.read_csv(config['load_path_ADL']+df['Folder'].values[i]+'/ag/agAll_DandTime_resampling.csv',header=None)
        label_table =  pd.read_csv(config['load_path_ADL']+'Labeling_final/'+df['Folder'].values[i]+'.csv',sep=',')
        
        Start_time = label_table['#starttime'][0]
        End_time = label_table['#endtime'].iloc[-1]
        period = (data_r.iloc[:, 4].values >= Start_time) & (data_r.iloc[:, 4].values <= End_time)
        data_r[5] = np.zeros((data_r.shape[0]),dtype=np.int64);
        data_r.iloc[:,5] = -1;
        data_r.iloc[period,5] = 0;
        

        for ii in range(1,label_table.shape[0]-1):
            if label_table['all_tiers'][ii] == 'walking':
                Start_label = label_table['#starttime'][ii]
                End_label = label_table['#endtime'][ii]
                period_label = (data_r.iloc[:, 4].values >= Start_label) & (data_r.iloc[:, 4].values <= End_label)
                data_r.iloc[period_label,5] = 1
            elif label_table['all_tiers'][ii] == 'ignore':
                Start_label = label_table['#starttime'][ii]
                End_label = label_table['#endtime'][ii]
                period_label = (data_r.iloc[:, 4].values >= Start_label) & (data_r.iloc[:, 4].values <= End_label)
                data_r.iloc[period_label,5] = -1
            
        c_data = data_r.loc[period,0:2].reset_index(drop=True)
        c_data_label = data_r.loc[period,5]
        c_Window_list, c_label1_list = SlidindWindow_and_lables_ADL(config,c_data, c_data_label) 
        c_subjectID_list = np.zeros((np.size(c_Window_list,0)),dtype=np.int64)
        c_subjectID_list[:] =  df['subjID'].iloc[i]
        
        Window_list = np.concatenate((Window_list,c_Window_list))
        label1_list = np.concatenate((label1_list,c_label1_list))
        subjectID_list = np.concatenate((subjectID_list,c_subjectID_list))
        
        
    # delete the 1st row
    Window_list = np.delete(Window_list, 0, axis=0)
    label1_list = np.delete(label1_list, 0, axis=0)
    subjectID_list = np.delete(subjectID_list, 0, axis=0)
    
    
    return Window_list, label1_list, subjectID_list


def NewLabel(config, c_numlabel_conti):
    new_label1 = np.zeros(np.shape(c_numlabel_conti)[0])
    new_label1[:] = -1
    new_label1[(c_numlabel_conti.iloc[:,0]<5) & (c_numlabel_conti.iloc[:,0]>-1)] = 1
    new_label1[(c_numlabel_conti.iloc[:,0]==5) | (c_numlabel_conti.iloc[:,0]==6)] = 2
    new_label1[(c_numlabel_conti.iloc[:,0]==7) | (c_numlabel_conti.iloc[:,0]==8)] = 3
    new_label1[(c_numlabel_conti.iloc[:,0]>8) ] = 0
    c_numlabel_conti_new = pd.concat([c_numlabel_conti, pd.DataFrame(new_label1.astype('int64'))],axis =1)
    new_label2 = np.zeros(np.shape(c_numlabel_conti)[0])
    new_label2[:] = -1
    new_label2[(c_numlabel_conti.iloc[:,0]<4) & (c_numlabel_conti.iloc[:,0]>-1)] = 1
    new_label2[(c_numlabel_conti.iloc[:,0]==4) & (c_numlabel_conti.iloc[:,0]>-1)] = 2
    new_label2[(c_numlabel_conti.iloc[:,0]==5) | (c_numlabel_conti.iloc[:,0]==6)] = 3
    new_label2[(c_numlabel_conti.iloc[:,0]==7) | (c_numlabel_conti.iloc[:,0]==8)] = 4
    new_label2[(c_numlabel_conti.iloc[:,0]>8) ] = 0
    c_numlabel_conti_new = pd.concat([c_numlabel_conti_new, pd.DataFrame(new_label2.astype('int64'))],axis =1)
    return c_numlabel_conti_new
