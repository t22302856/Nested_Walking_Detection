#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:08:31 2023

@author: kai-chunliu
"""
import pandas as pd
import numpy as np
import data_preprocessing
import ModelTrainTest


def LSOCV(df_r, df_ADL, config): #leave subjects out cross validation

    #list the unique subject index
    subject_ind=df_ADL["subjID"].unique()
    Performance_subject = pd.DataFrame([])
    
    Performance_ind=0    
    for Performance_ind in range(len(subject_ind)): 
        config['c_subject'] = subject_ind[Performance_ind]

        #split data group
        df_ADL_test = df_ADL[df_ADL['subjID'] == config['c_subject']]
        df_ADL_train = df_ADL[df_ADL['subjID'] != config['c_subject']]
        df_r_test = df_r[df_r['subjID'] == config['c_subject']]
        df_r_train = df_r[df_r['subjID'] != config['c_subject']]
        
        
        ## windowing & labeling ##
        #ADL data preparation
        # x_ADL, y_ADL_1, SubjIDList = data_preprocessing.create_segments_and_labels_ADL(config,df_ADL.loc[0:3,:])
        x_ADL, y_ADL_1, SubjIDList = data_preprocessing.create_segments_and_labels_ADL(config,df_ADL)
        
        # ADL test
        ADL_test_index = SubjIDList == config['c_subject']
        x_ADL_test = x_ADL[ADL_test_index]
        y_ADL_test1 =  y_ADL_1[ADL_test_index]
        SubjIDList_ADL_test =  SubjIDList[ADL_test_index]
        
        # ADL traing
        if config['ADL_train']:
            ADL_test_index = SubjIDList == config['c_subject']
            x_ADL_train = x_ADL[~ADL_test_index]
            y_ADL_train1 =  y_ADL_1[~ADL_test_index]
            SubjIDList_ADL_train =  SubjIDList[~ADL_test_index]
        else:
            ADL_test_index = SubjIDList == config['c_subject']
            x_ADL_train = []
            y_ADL_train1 =  []
            SubjIDList_ADL_train =  []
        
        # in-lab train
        if config['SpecifiedActivity_train']:
            x_r, y_r_1, y_r_2, SubjIDList = data_preprocessing.create_segments_and_labels(config,df_r.loc[:])
            r_test_index = SubjIDList == config['c_subject']
            x_r_train = x_r[~r_test_index]
            y_r_train2 = y_r_2[~r_test_index]
            SubjIDList_r_train = SubjIDList[~r_test_index] 
        else:
            x_r_train = []
            y_r_train2 = []
            SubjIDList_r_train = []

        # training and testing
#%% for CNN
        if config['ModelName'] == 'CNNc3f1':
            
            # 1. listing the training and validation performance of all parameters
            nested_perform_table =pd.DataFrame([])       
            for filter_width in config['filterSizeList']:
                for conv_filter in config['conv_filterList']:
                    for fc_filter in config['fc_filterList']:
                        config['filter_width'] = filter_width # filter size
                        config['conv_filters'] = conv_filter # number of filters for convolution layers
                        config['fc_filters'] = fc_filter # number of filter for the fully-connection layer
                        c_perform = ModelTrainTest.ModelTrain_train_vali(config, x_ADL_train, y_ADL_train1, SubjIDList_ADL_train, x_r_train, y_r_train2, SubjIDList_r_train)
                        c_parameter = pd.DataFrame([filter_width, conv_filter, fc_filter],index = ['filter_width', 'conv_filter', 'fc_filter']).T
                        c_nested_perform = pd.concat([c_parameter, c_perform],axis=1)
                        nested_perform_table = pd.concat([nested_perform_table, c_nested_perform])

            # 2. find the parameters with the best validation f1-score
            # reset the index for the nested CV 
            nested_perform_table =nested_perform_table.reset_index(drop=True)
            # find the parameters with the best validation f1-score
            nested_perform_best = pd.DataFrame(nested_perform_table.loc[nested_perform_table['vali_f1'].idxmax(),:]).T
            config['filter_width'], config['conv_filters'], config['fc_filters'], config['retrain_epoch'] = int(nested_perform_best['filter_width'].iloc[0]), int(nested_perform_best['conv_filter'].iloc[0]), int(nested_perform_best['fc_filter'].iloc[0]), int(nested_perform_best['best_epoch'].iloc[0])
            
            # 3. retraining the model with the best hyperparameters & testing
            perform_retrain_test, WalkTime = ModelTrainTest.ModelRetrain_test(config, x_ADL_train, y_ADL_train1, SubjIDList_ADL_train, x_r_train, y_r_train2, SubjIDList_r_train, 
                                                         x_ADL_test, y_ADL_test1, SubjIDList_ADL_test)
            
            perform_retrain_test['subjID'] = config['c_subject']
            Performance_subject = pd.concat([Performance_subject, perform_retrain_test])
            
            fileName = f"{config['output_path']}/nested_perform_table_{config['c_subject']}.csv"
            nested_perform_table.to_csv(fileName)
            fileName = f"{config['output_path']}/nested_perform_best_{config['c_subject']}.csv"
            nested_perform_best.to_csv(fileName)
            # fileName = f"{config['output_path']}/perform_retrain_test_{config['c_subject']}.csv"
            # perform_retrain_test.to_csv(fileName)
#%% for ResNet         
        if config['ModelName'] == 'ResNet':
            
            nested_perform_table =pd.DataFrame([])
            for fc_filter in config['fc_filterList']:
                config['fc_filters'] = fc_filter
                c_perform = ModelTrainTest.ModelTrain_train_vali(config, x_ADL_train, y_ADL_train1, SubjIDList_ADL_train, x_r_train, y_r_train2, SubjIDList_r_train)
                c_parameter = pd.DataFrame([fc_filter],index = ['fc_filter']).T
                c_nested_perform = pd.concat([c_parameter, c_perform],axis=1)
                nested_perform_table = pd.concat([nested_perform_table, c_nested_perform])
            
            # 2. retraining the model with the best hyperparameters
            # reset the index for the nested CV 
            nested_perform_table =nested_perform_table.reset_index(drop=True)
            # find the parameters with the best validation f1-score
            nested_perform_best = pd.DataFrame(nested_perform_table.loc[nested_perform_table['vali_f1'].idxmax(),:]).T
            config['fc_filters'], config['retrain_epoch'] = int(nested_perform_best['fc_filter'].iloc[0]), int(nested_perform_best['best_epoch'].iloc[0])
            
            # 3. testing
            perform_retrain_test, WalkTime = ModelTrainTest.ModelRetrain_test(config, x_ADL_train, y_ADL_train1, SubjIDList_ADL_train, x_r_train, y_r_train2, SubjIDList_r_train, 
                                                         x_ADL_test, y_ADL_test1, SubjIDList_ADL_test)
            
            perform_retrain_test['subjID'] = config['c_subject']
            Performance_subject = pd.concat([Performance_subject, perform_retrain_test])
            
            fileName = f"{config['output_path']}/nested_perform_table_{config['c_subject']}.csv"
            nested_perform_table.to_csv(fileName)
            fileName = f"{config['output_path']}/nested_perform_best_{config['c_subject']}.csv"
            nested_perform_best.to_csv(fileName)
    
    # # save the performance of each subject
    # fileName = f"{config['output_path']}/Performance_subject.csv"
    # Performance_subject.to_csv(fileName)
#%% calculate ovall performance and save
    Performance_average = pd.DataFrame(Performance_subject.mean()[:-1]).T
    Performance_average['subjID'] = 'overall'
    Performance_overall = pd.concat([Performance_subject, Performance_average])
    fileName = f"{config['output_path']}/Performance_overall.csv"
    Performance_overall.to_csv(fileName)
    
                    