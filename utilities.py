#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:16:49 2023

@author: kai-chunliu
"""

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches as pc
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import numpy as np
import os
from scipy import stats
import pandas as pd

import torch
from torch.utils.data import DataLoader
import umap


def signalPlot(c_data, label, subject,n=0):

    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(c_data)+1),c_data[:,0], label='x-axis Acc(m/s^2)')
    plt.plot(range(1,len(c_data)+1),c_data[:,1], label='y-axis Acc(m/s^2)')
    plt.plot(range(1,len(c_data)+1),c_data[:,2], label='z-axis Acc(m/s^2)')
    plt.xlabel('Data Point')
    # plt.grid(True)
    plt.legend(loc='upper left')
    plt.ylim(-3,3)
    plt.title(label+'_S'+subject+'_W'+str(n), fontsize = 20)
    plt.show()

def PerformanceSave(config, Performance, Performance_subject_df, Performance_cont, confu_AllFold, confu_conti_AllFold, WalkTime_subject):
    indexPerformance = [str(i) for i in range(config['fold'])]
    indexPerformance.append('overall')

    Performance_df=pd.DataFrame(Performance, index= indexPerformance, columns=['te_acc','te_sen','te_pre','te_f1', 'te_roc','tr_acc','tr_sen','tr_pre','tr_f1', 'tr_roc','vali_acc','vali_sen','vali_pre','vali_f1','vali_roc'])    
    fileName = f"{config['output_path']}WindowResults_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    Performance_df.to_csv(fileName, index = True, header=True)

    fileName = f"{config['output_path']}WindowResultsSubject_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    Performance_subject_df.to_csv(fileName, index = True, header=True)
    
    fileName = f"{config['output_path']}WalkTimeSubject_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    WalkTime_subject.to_csv(fileName, index = True, header=True)

    Performance_cont_df=pd.DataFrame(Performance_cont, index= indexPerformance, columns=['te_acc','te_sen','te_spe','te_pre'])    
    fileName = f"{config['output_path']}ContinuResults_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    Performance_cont_df.to_csv(fileName, index = True, header=True)

    if config['LabelTarget'] == 1:
        confu_AllFold_df = pd.DataFrame(confu_AllFold, index=['Walking', 'Ramp', 'Staris', 'Non-Walking'], columns=['Walking', 'Ramp', 'Staris', 'Non-Walking'])
        confu_conti_AllFold_df = pd.DataFrame(confu_conti_AllFold, index=['Walking', 'Ramp', 'Staris', 'Non-Walking'], columns=['Walking', 'Ramp', 'Staris', 'Non-Walking'])
    elif config['LabelTarget'] == 2:
        confu_AllFold_df = pd.DataFrame(confu_AllFold, index=['Walking', 'Non-Walking'], columns=['Walking', 'Non-Walking'])
        confu_conti_AllFold_df = pd.DataFrame(confu_conti_AllFold, index=['Walking', 'Non-Walking'], columns=['Walking', 'Non-Walking'])
    elif config['LabelTarget'] == 3:
        ID = ['sidewalk','uneven', 'terrain', 'tilted', 'walking', 'ramp', 'stairs','non-walking']
        confu_AllFold_df = pd.DataFrame(confu_AllFold, index=ID, columns=ID)
        confu_conti_AllFold_df = pd.DataFrame(confu_conti_AllFold, index=ID, columns=ID)
    elif config['LabelTarget'] == 4:
        ID = ['OutdoorWalking', 'IndoorWalking', 'Ramp', 'Staris', 'Non-Walking']
        confu_AllFold_df = pd.DataFrame(confu_AllFold, index=ID, columns=ID)
        confu_conti_AllFold_df = pd.DataFrame(confu_conti_AllFold, index=ID, columns=ID)
    elif config['LabelTarget'] == 5:
        ID = ['OutdoorWalking', 'IndoorWalking', 'Staris', 'Non-Walking']
        confu_AllFold_df = pd.DataFrame(confu_AllFold, index=ID, columns=ID)
        confu_conti_AllFold_df = pd.DataFrame(confu_conti_AllFold, index=ID, columns=ID)
        
    fileName = f"{config['output_path']}confu_AllFold_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    confu_AllFold_df.to_csv(fileName, index = True, header=True)
    fileName = f"{config['output_path']}confu_conti_AllFold_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    confu_conti_AllFold_df.to_csv(fileName, index = True, header=True)

def PerformanceSave_arg(config, Performance, Performance_subject_df, Performance_cont, confu_AllFold, confu_conti_AllFold):
    indexPerformance = list([config['cur_fold']])
    indexPerformance.append('overall')

    Performance_df=pd.DataFrame(Performance, index= indexPerformance, columns=['te_acc','te_sen','te_pre','te_f1', 'te_roc','tr_acc','tr_sen','tr_pre','tr_f1', 'tr_roc','vali_acc','vali_sen','vali_pre','vali_f1','vali_roc'])    
    fileName = f"{config['output_path']}WindowResults_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    Performance_df.to_csv(fileName, index = True, header=True)

    fileName = f"{config['output_path']}WindowResultsSubject_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    Performance_subject_df.to_csv(fileName, index = True, header=True)

    Performance_cont_df=pd.DataFrame(Performance_cont, index= indexPerformance, columns=['te_acc','te_sen','te_spe','te_pre'])    
    fileName = f"{config['output_path']}ContinuResults_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    Performance_cont_df.to_csv(fileName, index = True, header=True)

    if config['LabelTarget'] == 1:
        confu_AllFold_df = pd.DataFrame(confu_AllFold, index=['Walking', 'Ramp', 'Staris', 'Non-Walking'], columns=['Walking', 'Ramp', 'Staris', 'Non-Walking'])
        confu_conti_AllFold_df = pd.DataFrame(confu_conti_AllFold, index=['Walking', 'Ramp', 'Staris', 'Non-Walking'], columns=['Walking', 'Ramp', 'Staris', 'Non-Walking'])
    elif config['LabelTarget'] == 2:
        confu_AllFold_df = pd.DataFrame(confu_AllFold, index=['Walking', 'Non-Walking'], columns=['Walking', 'Non-Walking'])
        confu_conti_AllFold_df = pd.DataFrame(confu_conti_AllFold, index=['Walking', 'Non-Walking'], columns=['Walking', 'Non-Walking'])
    elif config['LabelTarget'] == 3:
        ID = ['sidewalk','uneven', 'terrain', 'tilted', 'walking', 'ramp', 'stairs','non-walking']
        confu_AllFold_df = pd.DataFrame(confu_AllFold, index=ID, columns=ID)
        confu_conti_AllFold_df = pd.DataFrame(confu_conti_AllFold, index=ID, columns=ID)
    elif config['LabelTarget'] == 4:
        ID = ['OutdoorWalking', 'IndoorWalking', 'Ramp', 'Staris', 'Non-Walking']
        confu_AllFold_df = pd.DataFrame(confu_AllFold, index=ID, columns=ID)
        confu_conti_AllFold_df = pd.DataFrame(confu_conti_AllFold, index=ID, columns=ID)
    elif config['LabelTarget'] == 5:
        ID = ['OutdoorWalking', 'IndoorWalking', 'Staris', 'Non-Walking']
        confu_AllFold_df = pd.DataFrame(confu_AllFold, index=ID, columns=ID)
        confu_conti_AllFold_df = pd.DataFrame(confu_conti_AllFold, index=ID, columns=ID)
        
    fileName = f"{config['output_path']}confu_AllFold_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    confu_AllFold_df.to_csv(fileName, index = True, header=True)
    fileName = f"{config['output_path']}confu_conti_AllFold_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.csv"
    confu_conti_AllFold_df.to_csv(fileName, index = True, header=True)
    
    
def ConfusionSave(config,GT,pred,subpath, fileName):
    if config['LabelTarget'] == 1:
        confu = confusion_matrix(GT,pred, labels=[1,2,3,0])
        confu_df = pd.DataFrame(confu, index=['Walking', 'Ramp', 'Staris', 'Non-Walking'], columns=['Walking', 'Ramp', 'Staris', 'Non-Walking'])        
    elif config['LabelTarget'] == 2:
        confu = confusion_matrix(GT,pred, labels=[1,0])
        confu_df = pd.DataFrame(confu, index=['Walking', 'Non-Walking'], columns=['Walking', 'Non-Walking'])
    elif config['LabelTarget'] == 3:
        confu = confusion_matrix(GT,pred, labels=[1,2,3,4,5,6,7,8,9,0])
        ID = ['sidewalk','uneven', 'terrain', 'tilted', 'walking', 'ramp', 'stairs','non-walking']
        confu_df = pd.DataFrame(confu, index=ID, columns=ID)
    elif config['LabelTarget'] == 4:
        confu = confusion_matrix(GT,pred, labels=[1,2,3,4,0])
        confu_df = pd.DataFrame(confu, index=['OutdoorWalking', 'IndoorWalking', 'Ramp', 'Staris', 'Non-Walking'], columns=['OutdoorWalking', 'IndoorWalking', 'Ramp', 'Staris', 'Non-Walking'])
    elif config['LabelTarget'] == 5:
        confu = confusion_matrix(GT,pred, labels=[1,2,3,0])
        ID = ['OutdoorWalking', 'IndoorWalking', 'Staris', 'Non-Walking']
        confu_df = pd.DataFrame(confu, index=ID, columns=ID)        
    
    try:
        os.mkdir(config['output_path']+subpath)
    except:
        print(config['output_path']+subpath+'is exst')
    SaveName = f"{config['output_path']}/{subpath}/{fileName}"
    confu_df.to_csv(SaveName, index = True, header=True)
    
    return confu
    

def plot2(train_loss, valid_loss, test_loss, acc_train, acc_vali, acc_test, f1_train, f1_vali, f1_test, config):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
    plt.plot(range(1,len(valid_loss)+1),test_loss,label='Test Loss')
    # plt.plot(range(1,len(acc_train)+1),acc_train, label='Training Acc')
    # plt.plot(range(1,len(acc_vali)+1),acc_vali, label='Validation Acc')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    # plt.yscale("log")
    # plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    try:
        os.mkdir(config['output_path']+'/LossCurve/')
    except:
        print(config['output_path']+'/LossCurve/'+'is exst')
    fileName = f"{config['output_path']}/LossCurve/LossPlot_{config['fold']}fold{config['cur_fold']}.png"
    fig.savefig(fileName, bbox_inches='tight')

    # visualize the Accuracy as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(acc_train)+1),acc_train, label='Training Acc')
    plt.plot(range(1,len(acc_vali)+1),acc_vali, label='Validation Acc')
    plt.plot(range(1,len(acc_test)+1),acc_test, label='Testing Acc')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    # plt.ylabel('loss')
    plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    
    fileName = f"{config['output_path']}/LossCurve/AccPlot_{config['fold']}fold{config['cur_fold']}.png"
    fig.savefig(fileName, bbox_inches='tight')
    
    # visualize the f1-scroe as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(f1_train)+1),f1_train, label='Training f1')
    plt.plot(range(1,len(f1_vali)+1),f1_vali, label='Validation f1')
    plt.plot(range(1,len(f1_test)+1),f1_test, label='Testing f1')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    # plt.ylabel('loss')
    plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    
    fileName = f"{config['output_path']}/LossCurve/f1Plot_{config['fold']}fold{config['cur_fold']}.png"
    fig.savefig(fileName, bbox_inches='tight')
    
def plot_train_vali_nestedCV(train_loss, valid_loss, acc_train, acc_vali, f1_train, f1_vali, config):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
    # plt.plot(range(1,len(valid_loss)+1),test_loss,label='Test Loss')
    # plt.plot(range(1,len(acc_train)+1),acc_train, label='Training Acc')
    # plt.plot(range(1,len(acc_vali)+1),acc_vali, label='Validation Acc')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    # plt.yscale("log")
    # plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    try:
        os.mkdir(config['output_path']+'/LossCurve/')
    except:
        print(config['output_path']+'/LossCurve/'+'is exst')
    # fileName = f"{config['output_path']}/LossCurve/LossPlot_{config['fold']}fold{config['cur_fold']}.png"
    # fig.savefig(fileName, bbox_inches='tight')

    # visualize the Accuracy as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(acc_train)+1),acc_train, label='Training Acc')
    plt.plot(range(1,len(acc_vali)+1),acc_vali, label='Validation Acc')
    # plt.plot(range(1,len(acc_test)+1),acc_test, label='Testing Acc')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    # plt.ylabel('loss')
    plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    
    # fileName = f"{config['output_path']}/LossCurve/AccPlot_{config['fold']}fold{config['cur_fold']}.png"
    # fig.savefig(fileName, bbox_inches='tight')
    
    # visualize the f1-scroe as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(f1_train)+1),f1_train, label='Training f1')
    plt.plot(range(1,len(f1_vali)+1),f1_vali, label='Validation f1')
    # plt.plot(range(1,len(f1_test)+1),f1_test, label='Testing f1')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    # plt.ylabel('loss')
    plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    
    # fileName = f"{config['output_path']}/LossCurve_nested/f1Plot_{config['fold']}fold{config['cur_fold']}.png"
    # fig.savefig(fileName, bbox_inches='tight')    

def results_plot(pred, label, config, c_subjectID, c_test_data, Performance, fileName='fig.png'):
    sen_str = str(round(Performance[1]*100,1))
    pre_str = str(round(Performance[1]*100,2))
    f1_str = str(round(Performance[1]*100,3))
    pred1 = pred[:]-5.8
    label1 = label[:]-5.5

    #acceleration data plot
    c_test_data1 = c_test_data[:]
    fig = plt.figure(figsize=(16,8))
    plt.plot(range(1,len(c_test_data1)+1),c_test_data1.iloc[:,0].values, label='x-axis Acc(m/s^2)')
    plt.plot(range(1,len(c_test_data1)+1),c_test_data1.iloc[:,1].values, label='x-axis Acc(m/s^2)')
    plt.plot(range(1,len(c_test_data1)+1),c_test_data1.iloc[:,2].values, label='x-axis Acc(m/s^2)')
    plt.plot(range(1,len(c_test_data1)+1),pred1, label='predict')
    plt.plot(range(1,len(c_test_data1)+1),label1, label='label')
    plt.xlabel('Samples')
    plt.title(f"{str(c_subjectID)}_Sen{sen_str}_Pre{pre_str}_F1{f1_str}")
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    try:
        os.mkdir(config['output_path']+'/ResultsPlot/')
    except:
        print(config['output_path']+'/ResultsPlot/'+'is exst')
    
    fileName = f"{config['output_path']}/ResultsPlot/{str(c_subjectID)}.png"
    # plt.show()
    fig.savefig(fileName, bbox_inches='tight',dpi=600)
    

    
    # try:
    #     os.mkdir(config['output_path']+'/ResultsPlot/')
    # except:
    #     print(config['output_path']+'/ResultsPlot/'+'is exst')
    # fileName = f"{config['output_path']}/ResultsPlot/AccelerationPlot_{c_subjectID}.png"
    # # plt.show()
    # fig.savefig(fileName, bbox_inches='tight',dpi=600)
    
    # #=============label and output plot===============
    # cm1 = plt.get_cmap('Accent')
    # cm3 = plt.get_cmap('tab20b')


    # pred1 = pred[0:]
    # # pred1[3000:6000] =1
    
    # fig = plt.figure(figsize=(16,8))
    # ax1 = plt.axes()
    # for i in range(len(pred1)):
    #     c_pred = pred1[i] 
    #     if c_pred==0:
    #         plt.fill([i,i,i+1,i+1],[0,1,1,0],color=cm1(2), linewidth=1)
    #     elif c_pred==1:
    #         plt.fill([i,i,i+1,i+1],[0,1,1,0],color=cm1(0), linewidth=1)
    #     else:
    #         plt.fill([i,i,i+1,i+1],[0,1,1,0],color='w', linewidth=1)
            
    #     c_label = label.iloc[i,1]
    #     if c_label==0:
    #         plt.fill([i,i,i+1,i+1],[-1,0,0,-1],color=cm1(2), linewidth=1)
    #     elif c_label==1:
    #         plt.fill([i,i,i+1,i+1],[-1,0,0,-1],color=cm1(0), linewidth=1)
    #     else:
    #         plt.fill([i,i,i+1,i+1],[-1,0,0,-1],color='w', linewidth=1)
        
    #     if c_label != -1:
    #         if (c_pred == c_label) & (c_pred == 1): #TP
    #             plt.fill([i,i,i+1,i+1],[-2,-1,-1,-2],color='k', linewidth=1)
    #         elif (c_pred == c_label) & (c_pred == 0): #TN
    #             plt.fill([i,i,i+1,i+1],[-2,-1,-1,-2],color='b', linewidth=1)
    #         elif (c_pred != c_label) & (c_pred == 1): #FP
    #             plt.fill([i,i,i+1,i+1],[-2,-1,-1,-2],color='r', linewidth=1)
    #         else: # FN
    #             plt.fill([i,i,i+1,i+1],[-2,-1,-1,-2],color='purple', linewidth=1)
    #         plt.fill([i,i,i+1,i+1],[-3,-2,-2,-3],color=cm3(label.iloc[i,0]), linewidth=1)
        

    # SepLine1 = np.zeros((len(pred1)))
    # SepLine2 = np.zeros((len(pred1)))
    # SepLine2[:] = -1
    # SepLine3 = np.zeros((len(pred1)))
    # SepLine3[:] = -2
    # plt.plot(range(0,len(pred1)),SepLine1,color='w',linewidth=2)
    # plt.plot(range(0,len(pred1)),SepLine2,color='w',linewidth=2)
    # plt.plot(range(0,len(pred1)),SepLine3,color='w',linewidth=2)
    # ax1.axes.get_yaxis().set_visible(False)
    
   
        
def results_post_plot(pred, label, config, c_subjectID, fileName='fig.png'):
    
    cm1 = plt.get_cmap('Accent')
    cm3 = plt.get_cmap('tab20b')


    pred1 = pred[0:]
    # pred1[3000:6000] =1
    
    fig = plt.figure(figsize=(16,8))
    ax1 = plt.axes()
    for i in range(len(pred1)):
        c_pred = pred1[i] 
        if c_pred==0:
            plt.fill([i,i,i+1,i+1],[0,1,1,0],color=cm1(2), linewidth=1)
        elif c_pred==1:
            plt.fill([i,i,i+1,i+1],[0,1,1,0],color=cm1(0), linewidth=1)
        else:
            plt.fill([i,i,i+1,i+1],[0,1,1,0],color='w', linewidth=1)
            
        c_label = label.iloc[i,1]
        if c_label==0:
            plt.fill([i,i,i+1,i+1],[-1,0,0,-1],color=cm1(2), linewidth=1)
        elif c_label==1:
            plt.fill([i,i,i+1,i+1],[-1,0,0,-1],color=cm1(0), linewidth=1)
        else:
            plt.fill([i,i,i+1,i+1],[-1,0,0,-1],color='w', linewidth=1)
        
        if c_label != -1:
            if (c_pred == c_label) & (c_pred == 1): #TP
                plt.fill([i,i,i+1,i+1],[-2,-1,-1,-2],color='k', linewidth=1)
            elif (c_pred == c_label) & (c_pred == 0): #TN
                plt.fill([i,i,i+1,i+1],[-2,-1,-1,-2],color='b', linewidth=1)
            elif (c_pred != c_label) & (c_pred == 1): #FP
                plt.fill([i,i,i+1,i+1],[-2,-1,-1,-2],color='r', linewidth=1)
            else: # FN
                plt.fill([i,i,i+1,i+1],[-2,-1,-1,-2],color='purple', linewidth=1)
            plt.fill([i,i,i+1,i+1],[-3,-2,-2,-3],color=cm3(label.iloc[i,0]), linewidth=1)
        

    SepLine1 = np.zeros((len(pred1)))
    SepLine2 = np.zeros((len(pred1)))
    SepLine2[:] = -1
    SepLine3 = np.zeros((len(pred1)))
    SepLine3[:] = -2
    plt.plot(range(0,len(pred1)),SepLine1,color='w',linewidth=2)
    plt.plot(range(0,len(pred1)),SepLine2,color='w',linewidth=2)
    plt.plot(range(0,len(pred1)),SepLine3,color='w',linewidth=2)
    ax1.axes.get_yaxis().set_visible(False)
    
    try:
        os.mkdir(config['output_path']+'/ResultsPlot/')
    except:
        print(config['output_path']+'/ResultsPlot/'+'is exst')
    fileName = f"{config['output_path']}/ResultsPlot/ResultsPlot_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}_{c_subjectID}_post.png"
    
    # plt.show()
    fig.savefig(fileName, bbox_inches='tight',dpi=600)    

def plot_loss(train_losses,test_losses, fold, floag,config):
    plt.figure(figsize=(10,5))
    plt.title("Loss Curve")
    plt.plot(train_losses,label="train")
    plt.plot(test_losses,label="test")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig('books_read.png')

        
def show_confusion_matrix(validations, predictions, LABELS,config):

    matrix = confusion_matrix(validations, predictions)
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    fileName = f"{config['output_path']}Confusion_WS{int(config['window_size']/30)}_CNNfilterNumber{config['conv_filters']}_CNNfilterWidth{config['filter_width']}_FCfilterNum{config['fc_filters']}.png"
    fig.savefig(fileName, bbox_inches='tight')
    
def postprocessing(test_epoch_preds,config):
    post_test_epoch_preds = test_epoch_preds
    m1 = config['post_parameter_walk']
    m2 = config['post_parameter_nonwalk']    
    for i in range(np.max((m1,m2)),(len(post_test_epoch_preds)-np.max((m1,m2)))):
        c_label = test_epoch_preds[i]
        
        if c_label ==1:
            c_seq = test_epoch_preds[i-m1:i+m1+1]
            Mod = stats.mode(c_seq)[0][0]
            if c_label != Mod:
                test_epoch_preds[i] = Mod            
        else:
            c_seq = test_epoch_preds[i-m1:i+m1+1]
            Mod = stats.mode(c_seq)[0][0]
            if c_label != Mod:
                test_epoch_preds[i] = Mod
        
    return post_test_epoch_preds  
    
def sequence_generate(test_epoch_preds,c_numlabel_conti,config):
    c_numlabel_conti = c_numlabel_conti.reset_index(drop =True)
    overlap = config['overlap']
    c_numpred_conti = np.zeros((np.size(c_numlabel_conti,0)),dtype=np.int8)
    c_numpred_conti[:] = -1
    for i in range(len(test_epoch_preds)):
        c_numpred_conti[i*overlap:(i+1)*overlap] = test_epoch_preds[i]
    c_numpred_conti[len(test_epoch_preds)*overlap:(len(test_epoch_preds)+1)*overlap] = test_epoch_preds[i]
    ignore_ind = c_numlabel_conti == -1
    c_numpred_conti[ignore_ind] = -1
    
    return c_numpred_conti
        
def FeatureVisualization(config, model, x_test, y_test, device):
    model.eval()
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    if config['ModelName'] == 'CNNc3f1':
        model.fc1.register_forward_hook(get_activation('fc1'))
        embedding_all =np.zeros((1,config['fc_filters']))
    elif config['ModelName'] == 'ResNet':
        model.classifier.linear1.register_forward_hook(get_activation('classifier.linear1'))
        embedding_all =np.zeros((1,config['fc_filters']))
        
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float().to(device), torch.from_numpy(y_test).to(device))
    testloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    
    test_epoch_preds = []
    test_epoch_gt = []
    
    with torch.no_grad():
        # iterate over the valloader object (it'll return batches which you can use)
        for ii, (x, y) in enumerate(testloader):
            # sends batch x and y to the GPU
            inputs, targets = x.to(device), y.to(device)

            # send inputs through network to get predictions
            test_output = model(inputs)

            
            # calculate actual predictions (i.e. softmax probabilites); use torch.nn.functional.softmax() on dim=1
            test_output = torch.nn.functional.softmax(test_output, dim=1)
            
            if device =='cpu':
                #extract embedding
                if config['ModelName'] == 'CNNc3f1':
                    embedding = activation['fc1'].numpy()
                elif config['ModelName'] == 'ResNet':
                    embedding = activation['classifier.linear1'].numpy()
                embedding_all = np.concatenate((embedding_all,embedding))
            else:
                #extract embedding
                if config['ModelName'] == 'CNNc3f1':
                    embedding = activation['fc1'].cpu().numpy()
                elif config['ModelName'] == 'ResNet':
                    embedding = activation['classifier.linear1'].cpu().numpy()
                embedding_all = np.concatenate((embedding_all,embedding))
                
            
            
            # creates predictions and true labels; appends them to the final lists
            y_preds = np.argmax(test_output.cpu().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            test_epoch_preds = np.concatenate((np.array(test_epoch_preds, int), np.array(y_preds, int)))
            test_epoch_gt = np.concatenate((np.array(test_epoch_gt, int), np.array(y_true, int)))

        embedding_all = embedding_all[1:]
        # UMAP to embedding
        reducer = umap.UMAP()
        embedding_all_scaled = StandardScaler().fit_transform(embedding_all)
        embedding_all_UMAP = reducer.fit_transform(embedding_all_scaled)
        embedding_all_UMAP.shape
        

        
        #plot feature
        Acc= round(accuracy_score(y_test, test_epoch_preds)*100,2)
        f1 = round(f1_score(y_test, test_epoch_preds, average=None, labels=np.array([0,1]))[1]*100,2)
        class_names= ['non-walking', 'walking']
        labels = [class_names[label] for label in y_test]
        fig, ax = plt.subplots(1, figsize=(14, 10))
        scatter = plt.scatter(embedding_all_UMAP[:, 0], embedding_all_UMAP[:, 1], s=30, c=y_test, label=labels, cmap='Spectral', alpha=0.6)
        # Create a legend manually using class names from the `class_names` list
        legend = ax.legend(*scatter.legend_elements(), title="Classes", framealpha=0.8, fontsize=18)
        ax.add_artist(legend)
        plt.title(f"Features_{Acc}ACC_{f1}F1", fontsize=24)
        try:
            os.mkdir(config['output_path']+'/UMAP/')
        except:
            print(config['output_path']+'/UMAP/'+'is exst')
        fileName = f"{config['output_path']}/UMAP/UMAP_{config['fold']}fold{config['cur_fold']}.png"
        
        plt.savefig(fileName, bbox_inches='tight',dpi=600)
        plt.show()
        
# def ActivityIndex(df, epoch=1, fs=30): # input is the dataframe involving x-axis, y-axis, z-axis data
#     epoch_len = epoch * fs
#     window_len = fs
    
#     AI_all = []
#     for i in range(0, len(df), epoch_len):
#         df_epoch = df.iloc[i:i + epoch_len]
#         AI_epoch = []
#         for ii in range(0, len(df_epoch), window_len):
#             df_window = df_epoch.iloc[ii:ii + window_len]
#             x_var = np.var(df_window.iloc[:,0])
#             y_var = np.var(df_window.iloc[:,1])
#             z_var = np.var(df_window.iloc[:,2])
    
#             mean_var = np.sqrt((x_var + y_var + z_var)/3)**2
#             AI_window = (((x_var-mean_var)/(mean_var))+                     
#                         ((y_var-mean_var)/(mean_var))+
#                         ((z_var-mean_var)/(mean_var)))
#             if AI_window >0:
#                 AI_epoch.append(np.sqrt(AI_window))
#             else:
#                 AI_epoch.append(0)
#         AI = sum(AI_epoch)
#         AI_all.append(AI)
#     return AI_all
    
        
    