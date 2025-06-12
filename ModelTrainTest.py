#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:59:58 2024

@author: kai-chunliu
"""
import pandas as pd
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report, roc_auc_score
from sklearn.utils import class_weight

import torch
from torch.utils.data import DataLoader
from torch import nn
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, KFold, StratifiedKFold


import data_preprocessing
from utilities import plot_loss, plot2, plot_train_vali_nestedCV, results_plot, results_post_plot, postprocessing, sequence_generate, ConfusionSave, FeatureVisualization
import models
from collections import Counter

import copy


device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

def get_class_weights(y):
    # obtain inverse of frequency as weights for the loss function
    counter = Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(counter)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)
    print("Weight tensor: ")
    print(weights)
    return weights

def load_weights(
    weight_path, model, my_device
):
    # only need to change weights name when
    # the model is trained in a distributed manner

    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names

    # distributed pretraining can be inferred from the keys' module. prefix
    head = next(iter(pretrained_dict_v2)).split('.')[0]  # get head of first key
    if head == 'module':
        # remove module. prefix from dict keys
        pretrained_dict_v2 = {k.partition('module.')[2]: pretrained_dict_v2[k] for k in pretrained_dict_v2.keys()}

    if hasattr(model, 'module'):
        model_dict = model.module.state_dict()
        multi_gpu_ft = True
    else:
        model_dict = model.state_dict()
        multi_gpu_ft = False

    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict_v2.items()
        if k in model_dict and k.split(".")[0] != "classifier"
    }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    if multi_gpu_ft:
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm1d") != -1:
        m.eval()
        
def freeze_weights(model):
    i = 0
    # Set Batch_norm running stats to be frozen
    # Only freezing ConV layers for now
    # or it will lead to bad results
    # http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
    for name, param in model.named_parameters():
        if name.split(".")[0] == "feature_extractor":
            param.requires_grad = False
            i += 1
    print("Weights being frozen: %d" % i)
    model.apply(set_bn_eval)

def init_model(config, device):
    if config['resnet_version'] > 0:
        model = models.Resnet(
            output_size=config['nb_classes'],
            is_eva=True,
            resnet_version=1,
            epoch_len=10,
            config=config
        )
    # else:
    #     model = SSLNET(
    #         output_size=cfg.data.output_size, flatten_size=1024
    #     )  # VGG

    # if cfg.multi_gpu:
    #     model = nn.DataParallel(model, device_ids=cfg.gpu_ids)

    model.to(device, dtype=torch.float)
    return model

def setup_model(config, my_device):
    model = init_model(config, device)

    if config['load_weights']:
        load_weights(
            config['flip_net_path'],
            model,
            my_device
        )
    if config['freeze_weight']:
        freeze_weights(model)
    return model

def train(model, trainloader, optimizer, criterion):
    # helper objects needed for proper documentation
    train_epoch_losses = []
    train_epoch_preds = []
    train_epoch_gt = []
    train_epoch_output = []
    start_time = time.time()

    # iterate over the trainloader object (it'll return batches which you can use)
    model.train()
    for i, (x, y) in enumerate(trainloader):
        # sends batch x and y to the GPU
        if x.size()[0]>1:
            inputs, targets = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # send inputs through network to get predictions
            train_output = model(inputs)
                        
            
            # calculates loss
            loss = criterion(train_output, targets.long())
            
            # backprogate your computed loss through the network
            # use the .backward() and .step() function on your loss and optimizer
            loss.backward()
            optimizer.step()
            
            # calculate actual predictions (i.e. softmax probabilites); use torch.nn.functional.softmax()
            train_output = torch.nn.functional.softmax(train_output, dim=1)
            
            # appends the computed batch loss to list
            train_epoch_losses.append(loss.item())
            
            
            # creates predictions and true labels; appends them to the final lists
            y_preds = np.argmax(train_output.cpu().detach().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            train_epoch_preds = np.concatenate((np.array(train_epoch_preds, int), np.array(y_preds, int)))
            train_epoch_gt = np.concatenate((np.array(train_epoch_gt, int), np.array(y_true, int)))
            train_epoch_output = np.concatenate((np.array(train_epoch_output, float), train_output.cpu().detach().numpy()[:,1]))

        
        
    elapsed = time.time() - start_time
    return train_epoch_losses, train_epoch_preds,train_epoch_gt, train_epoch_output, elapsed 

def validate(model, testloader, criterion):
    # helper objects
    test_epoch_preds = []
    test_epoch_gt = []
    test_epoch_losses = []
    test_epoch_output = []

    # sets network to eval mode and 
    model.eval()
    with torch.no_grad():
    # iterate over the valloader object (it'll return batches which you can use)
        for i, (x, y) in enumerate(testloader):
            # sends batch x and y to the GPU
            inputs, targets = x.to(device), y.to(device)

            # send inputs through network to get predictions
            test_output = model(inputs)

            # calculates loss by passing criterion both predictions and true labels 
            test_loss = criterion(test_output, targets.long())

            # calculate actual predictions (i.e. softmax probabilites); use torch.nn.functional.softmax() on dim=1
            test_output = torch.nn.functional.softmax(test_output, dim=1)

            # appends test loss to list
            test_epoch_losses.append(test_loss.item())

            # creates predictions and true labels; appends them to the final lists
            y_preds = np.argmax(test_output.cpu().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            test_epoch_preds = np.concatenate((np.array(test_epoch_preds, int), np.array(y_preds, int)))
            test_epoch_gt = np.concatenate((np.array(test_epoch_gt, int), np.array(y_true, int)))
            # test_epoch_output = np.concatenate((np.array(test_epoch_output, float), test_output.cpu().detach().numpy()[:,1]))
            test_epoch_output = np.concatenate((np.array(test_epoch_output, float), test_output.cpu().detach().numpy()[:,1]))
            
            
        return test_epoch_losses, test_epoch_preds,test_epoch_gt, test_epoch_output

def WalkingTimeEstimation(df,config):
    
    # find walking and non-walking segments
    arr = df
    segments = []
    current_value = arr[0]
    current_count = 1

    for num in arr[1:]:
        if num == current_value:
            current_count += 1
        else:
            segments.append((current_value, current_count))
            current_value = num
            current_count = 1
    segments.append((current_value, current_count))
    segments_df = np.array(segments) # 1st column is walking or non/walking, 2nd column is the number of window
    
    # find walking segments and transfer to walking time in seconds
    segments_df_walking = segments_df[segments_df[:,0]==1,:]
    segments_df_walking_time = config['window_size']/30 + (segments_df_walking[:,1] - 1)*config['overlap']/30
    TotalWalkingTime = sum(segments_df_walking_time)
    
    return TotalWalkingTime

def test_window(testloader, model, criterion, config):
    # helper objects
    test_epoch_preds = []
    test_epoch_gt = []
    test_epoch_output = []


    model.eval()
    with torch.no_grad():
    # iterate over the valloader object (it'll return batches which you can use)
        for i, (x, y) in enumerate(testloader):
            # sends batch x and y to the GPU
            inputs, targets = x.to(device), y.to(device)

            # send inputs through network to get predictions
            test_output = model(inputs)

            # calculate actual predictions (i.e. softmax probabilites); use torch.nn.functional.softmax() on dim=1
            test_output = torch.nn.functional.softmax(test_output, dim=1)


            # creates predictions and true labels; appends them to the final lists
            y_preds = np.argmax(test_output.cpu().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            test_epoch_preds = np.concatenate((np.array(test_epoch_preds, int), np.array(y_preds, int)))
            test_epoch_gt = np.concatenate((np.array(test_epoch_gt, int), np.array(y_true, int)))
            test_epoch_output = np.concatenate((np.array(test_epoch_output, float), test_output.detach().numpy()[:,1]))
    WalkTime_gt = WalkingTimeEstimation(test_epoch_gt,config)
    WalkTime_preds = WalkingTimeEstimation(test_epoch_preds,config)
    # WalkTime = np.array([WalkTime_gt, WalkTime_preds])
    WalkTime = pd.DataFrame([WalkTime_gt, WalkTime_preds],index=['gt_walktime','predic_walktime']).T
    

    return test_epoch_preds,test_epoch_gt, test_epoch_output, WalkTime

def ModelTrain_train_vali(config, x_ADL = [], y_ADL = [], SubjIDList_ADL = [], x_r = [], y_r_2= [], SubjIDList_r= []):
    
    subject_ind=np.unique(SubjIDList_ADL)
    Performance = pd.DataFrame(np.zeros((len(subject_ind),12),dtype=np.float32), 
                                       columns=['record_id','train_acc','train_sen','train_pre','train_f1', 'train_roc',
                                                'vali_acc','vali_sen','vali_pre','vali_f1', 'vali_roc', 'best_epoch'])
    Performance['record_id'] = subject_ind
    for Performance_ind in range(len(subject_ind)):
        vali_subject = subject_ind[Performance_ind]
        
        #split group
        # ADL validation
        ADL_vali_index = SubjIDList_ADL == vali_subject
        x_ADL_vali = x_ADL[ADL_vali_index]
        y_ADL_vali1 =  y_ADL[ADL_vali_index]
        SubjIDList_ADL_vali =  SubjIDList_ADL[ADL_vali_index]
        
        # ADL_training
        if config['ADL_train']:
            ADL_vali_index = SubjIDList_ADL == vali_subject
            x_ADL_train = x_ADL[~ADL_vali_index]
            y_ADL_train1 =  y_ADL[~ADL_vali_index]
            SubjIDList_ADL_train =  SubjIDList_ADL[~ADL_vali_index]
        
        # In-lab training
        if config['SpecifiedActivity_train']:
            r_vali_index = SubjIDList_r == vali_subject
            x_r_train = x_r[~r_vali_index]
            y_r_train2 = y_r_2[~r_vali_index]
            SubjIDList_r_train = SubjIDList_r[~r_vali_index] 
        
        # assign training and validation
        # train
        if config['SpecifiedActivity_train'] & config['ADL_train']:
            x_train = np.concatenate((x_r_train,x_ADL_train))
            y_train = np.concatenate((y_r_train2,y_ADL_train1))
        elif config['SpecifiedActivity_train']:
            x_train = x_r_train
            y_train = y_r_train2
        
        elif config['ADL_train']:
            x_train = x_ADL_train
            y_train = y_ADL_train1
        # test
        x_vali = x_ADL_vali
        y_vali = y_ADL_vali1
            
        if config['ModelName'] == 'ResNet':
            x_train = x_train.reshape((-1,config['nb_channels'],config['window_size']))
            x_vali = x_vali.reshape((-1,config['nb_channels'],config['window_size']))

        #for pytorch format
        # choose labels
        x_train, x_vali = x_train.astype('float32'), x_vali.astype('float32')

        
        config['nb_classes'] = len(np.unique(y_train))
        
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float().to(device), torch.from_numpy(y_train).to(device))
        vali_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_vali).float().to(device), torch.from_numpy(y_vali).to(device))
        
        
        trainloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)     
        valiloader = DataLoader(vali_dataset, batch_size=config['batch_size'], shuffle=False)     
        
        
        #impot model
        # network = DeepConvLSTM(config)
        if config['ModelName'] == 'CNNc2f1':
            model = models.CNNc2f1(config).to(device)
        elif config['ModelName'] == 'CNNc3f1':
            model = models.CNNc3f1(config).to(device)
        elif config['ModelName'] == 'ResNet':
            model = setup_model(config, device)
            
        
        
        # initialize the optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        if config['LoosFunction'] == 'weight':
            class_weights=class_weight.compute_class_weight('balanced',classes = np.unique(y_train),y = y_train)
            class_weights=torch.tensor(class_weights,dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean').to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
        

        
        # initialize the early_stopping object
        if config['early_stopping']:
            early_stopping = EarlyStopping(patience=config['patience'], verbose=True)
            # early_stopping = EarlyStopping(patience=config['patience'], path=config['output_path'] + f"checkpoint_nested_{config['c_subject']}.pt", verbose=False)
        
        loss_train, loss_vali = [], []
        acc_train, acc_vali = [], []
        f1_train, f1_vali= [], []
        roc_train, roc_vali = [], []
        
        
        # define your training loop; iterates over the number of epochs
        for e in range(config['epochs']):
            
            train_epoch_losses, train_epoch_preds,train_epoch_gt, train_epoch_output, elapsed = train(model, trainloader, optimizer, criterion)
            vali_epoch_losses, vali_epoch_preds,vali_epoch_gt, vali_epoch_output = validate(model, valiloader, criterion)
            # test_epoch_losses, test_epoch_preds,test_epoch_gt, test_epoch_output = validate(model, testloader, criterion)
            
            loss_train.append(np.mean(train_epoch_losses))
            loss_vali.append(np.mean(vali_epoch_losses))
            # loss_test.append(np.mean(test_epoch_losses))
            acc_train.append(accuracy_score(train_epoch_gt, train_epoch_preds))
            acc_vali.append(accuracy_score(vali_epoch_gt, vali_epoch_preds))
            # acc_test.append(accuracy_score(test_epoch_gt, test_epoch_preds))
            f1_train.append(f1_score(train_epoch_gt, train_epoch_preds, average='macro'))
            f1_vali.append(f1_score(vali_epoch_gt, vali_epoch_preds, average='macro'))
            # f1_test.append(f1_score(test_epoch_gt, test_epoch_preds, average='macro'))
            roc_train.append(roc_auc_score(train_epoch_gt, train_epoch_output, average='macro'))
            roc_vali.append(roc_auc_score(vali_epoch_gt, vali_epoch_output, average='macro'))
            # roc_test.append(roc_auc_score(test_epoch_gt, test_epoch_output, average='macro'))
            
            
            print("\nEPOCH: {}/{}".format(e + 1, config['epochs']),
                  "\n {:5.4f} s/epoch".format(elapsed),
                  "\nTrain Loss: {:.4f}".format(np.mean(train_epoch_losses)),
                  "Train Acc: {:.4f}".format(accuracy_score(train_epoch_gt, train_epoch_preds)),
                  # "Train Prec: {:.4f}".format(precision_score(train_epoch_gt, train_epoch_preds, average='macro')),
                  # "Train Rcll: {:.4f}".format(recall_score(train_epoch_gt, train_epoch_preds, average='macro')),
                   "Train F1: {:.4f}".format(f1_score(train_epoch_gt, train_epoch_preds, average=None)[-1]),
                  "\nVali Loss: {:.4f}".format(np.mean(vali_epoch_losses)),
                  "Vali Acc: {:.4f}".format(accuracy_score(vali_epoch_gt, vali_epoch_preds)),
                  # "Vali Prec: {:.4f}".format(precision_score(vali_epoch_gt, vali_epoch_preds, average='macro')),
                  # "Vali Rcll: {:.4f}".format(recall_score(vali_epoch_gt, vali_epoch_preds, average='macro')),
                   "Vali F1: {:.4f}".format(f1_score(vali_epoch_gt, vali_epoch_preds, average=None)[-1]))
                  # "\nTest Loss: {:.4f}".format(np.mean(test_epoch_losses)),
                  # "Test Acc: {:.4f}".format(accuracy_score(test_epoch_gt, test_epoch_preds)),
                  # "Test F1: {:.4f}".format(f1_score(test_epoch_gt, test_epoch_preds, average=None)[-1]))
            
            # save_best_model(np.mean(vali_epoch_losses), e, model, optimizer, criterion)
            # print('-'*50)
            if config['early_stopping']:
                early_stopping(np.mean(vali_epoch_losses), model) 
                # early_stopping(f1_score(vali_epoch_gt, vali_epoch_preds, average='macro')*-1, model) 
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                    
            
        # load the last checkpoint with the best model
        if config['early_stopping']:
            model.load_state_dict(torch.load(f"checkpoint.pt"))
            # model.load_state_dict(torch.load(config['output_path'] +f"checkpoint_nested_{config['c_subject']}.pt"))

        
        plot_train_vali_nestedCV(loss_train, loss_vali, acc_train, acc_vali, f1_train, f1_vali, config)
        
        
        #training
        train_epoch_losses, train_epoch_preds,train_epoch_gt, train_epoch_output = validate(model, trainloader, criterion)
        # confu_train = confusion_matrix(train_epoch_gt, train_epoch_preds)
        # confu_df = pd.DataFrame(confu_train, index=['Non-Walking', 'Walking', 'Ramp', 'Staris'], columns=['Non-Walking', 'Walking', 'Ramp', 'Staris'])
        Acc= accuracy_score(train_epoch_gt, train_epoch_preds)
        Sen= recall_score(train_epoch_gt, train_epoch_preds, average=None, labels=np.unique(train_epoch_gt))[-1]
        Pre= precision_score(train_epoch_gt, train_epoch_preds, average=None, labels=np.unique(train_epoch_gt))[-1]
        f1= f1_score(train_epoch_gt, train_epoch_preds, average=None, labels=np.unique(train_epoch_gt))[-1]
        roc = roc_auc_score(train_epoch_gt, train_epoch_output, average=None, labels=np.unique(train_epoch_gt))
        Performance.iloc[Performance_ind,1:6]=np.array([Acc, Sen, Pre, f1, roc], dtype=np.float32)  
        
        #validation
        vali_epoch_losses, vali_epoch_preds,vali_epoch_gt, vali_epoch_output  = validate(model, valiloader, criterion)
        # confu_vali = confusion_matrix(vali_epoch_gt, vali_epoch_preds)
        # confu_df = pd.DataFrame(confu_vali, index=['Non-Walking', 'Walking', 'Ramp', 'Staris'], columns=['Non-Walking', 'Walking', 'Ramp', 'Staris'])
        Acc= accuracy_score(vali_epoch_gt, vali_epoch_preds)
        Sen= recall_score(vali_epoch_gt, vali_epoch_preds, average=None, labels=np.unique(vali_epoch_gt))[-1]
        Pre= precision_score(vali_epoch_gt, vali_epoch_preds, average=None, labels=np.unique(vali_epoch_gt))[-1]
        f1= f1_score(vali_epoch_gt, vali_epoch_preds, average=None, labels=np.unique(vali_epoch_gt))[-1]
        roc = roc_auc_score(vali_epoch_gt, vali_epoch_output, average=None, labels=np.unique(vali_epoch_gt))
        Performance.iloc[Performance_ind,6:11]=np.array([Acc, Sen, Pre, f1, roc], dtype=np.float32)  
        
        #save current epoch
        Performance.iloc[Performance_ind,11]= e 
    Performance_ave = pd.DataFrame(Performance.mean()[1:]).T
    return  Performance_ave

def ModelRetrain_test(config, x_ADL_train = [], y_ADL_train1 = [], SubjIDList_ADL_train = [], x_r_train = [], y_r_train2= [], SubjIDList_r_train= [],
                 x_ADL_vali = [], y_ADL_vali1 = [], SubjIDList_ADL_vali = []):
    

    
    Performance = pd.DataFrame(np.zeros((1,10),dtype=np.float32), 
                                       columns=['train_acc','train_sen','train_pre','train_f1', 'train_roc',
                                                'test_acc','test_sen','test_pre','test_f1', 'test_roc'])

    # assign training and validation
    # train
    if config['SpecifiedActivity_train'] & config['ADL_train']:
        x_train = np.concatenate((x_r_train,x_ADL_train))
        y_train = np.concatenate((y_r_train2,y_ADL_train1))
    elif config['SpecifiedActivity_train']:
        x_train = x_r_train
        y_train = y_r_train2
    
    elif config['ADL_train']:
        x_train = x_ADL_train
        y_train = y_ADL_train1
    # test
    x_vali = x_ADL_vali
    y_vali = y_ADL_vali1
        
    
    if config['ModelName'] == 'ResNet':
        x_train = x_train.reshape((-1,config['nb_channels'],config['window_size']))
        x_vali = x_vali.reshape((-1,config['nb_channels'],config['window_size']))

    #for pytorch format
    # choose labels
    x_train, x_vali = x_train.astype('float32'), x_vali.astype('float32')

    
    config['nb_classes'] = len(np.unique(y_train))
    
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float().to(device), torch.from_numpy(y_train).to(device))
    vali_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_vali).float().to(device), torch.from_numpy(y_vali).to(device))
    
    
    trainloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)     
    valiloader = DataLoader(vali_dataset, batch_size=config['batch_size'], shuffle=False)     
    
    
    #impot model
    # network = DeepConvLSTM(config)
    if config['ModelName'] == 'CNNc2f1':
        model = models.CNNc2f1(config).to(device)
    elif config['ModelName'] == 'CNNc3f1':
        model = models.CNNc3f1(config).to(device)
    elif config['ModelName'] == 'ResNet':
        model = setup_model(config, device)
        
    
    
    # initialize the optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    if config['LoosFunction'] == 'weight':
        class_weights=class_weight.compute_class_weight('balanced',classes = np.unique(y_train),y = y_train)
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean').to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    

    
    # initialize the early_stopping object
    if config['early_stopping']:
        early_stopping = EarlyStopping(patience=config['patience'], verbose=True)
        # early_stopping = EarlyStopping(patience=config['patience'], path=config['output_path'] + f"checkpoint_retrain_{config['c_subject']}.pt", verbose=False)
    
    loss_train, loss_vali = [], []
    acc_train, acc_vali = [], []
    f1_train, f1_vali= [], []
    roc_train, roc_vali = [], []
    
    
    # define your training loop; iterates over the number of epochs
    for e in range(config['retrain_epoch']):
        
        train_epoch_losses, train_epoch_preds,train_epoch_gt, train_epoch_output, elapsed = train(model, trainloader, optimizer, criterion)
        vali_epoch_losses, vali_epoch_preds,vali_epoch_gt, vali_epoch_output = validate(model, valiloader, criterion)
        # test_epoch_losses, test_epoch_preds,test_epoch_gt, test_epoch_output = validate(model, testloader, criterion)
        
        loss_train.append(np.mean(train_epoch_losses))
        loss_vali.append(np.mean(vali_epoch_losses))
        # loss_test.append(np.mean(test_epoch_losses))
        acc_train.append(accuracy_score(train_epoch_gt, train_epoch_preds))
        acc_vali.append(accuracy_score(vali_epoch_gt, vali_epoch_preds))
        # acc_test.append(accuracy_score(test_epoch_gt, test_epoch_preds))
        f1_train.append(f1_score(train_epoch_gt, train_epoch_preds, average='macro'))
        f1_vali.append(f1_score(vali_epoch_gt, vali_epoch_preds, average='macro'))
        # f1_test.append(f1_score(test_epoch_gt, test_epoch_preds, average='macro'))
        # roc_train.append(roc_auc_score(train_epoch_gt, train_epoch_output, average='macro'))
        # roc_vali.append(roc_auc_score(vali_epoch_gt, vali_epoch_output, average='macro'))
        # roc_test.append(roc_auc_score(test_epoch_gt, test_epoch_output, average='macro'))
        
        
        print("\nEPOCH: {}/{}".format(e + 1, config['epochs']),
              "\n {:5.4f} s/epoch".format(elapsed),
              "\nTrain Loss: {:.4f}".format(np.mean(train_epoch_losses)),
              "Train Acc: {:.4f}".format(accuracy_score(train_epoch_gt, train_epoch_preds)),
              # "Train Prec: {:.4f}".format(precision_score(train_epoch_gt, train_epoch_preds, average='macro')),
              # "Train Rcll: {:.4f}".format(recall_score(train_epoch_gt, train_epoch_preds, average='macro')),
               "Train F1: {:.4f}".format(f1_score(train_epoch_gt, train_epoch_preds, average=None)[-1]),
              "\nVali Loss: {:.4f}".format(np.mean(vali_epoch_losses)),
              "Vali Acc: {:.4f}".format(accuracy_score(vali_epoch_gt, vali_epoch_preds)),
              # "Vali Prec: {:.4f}".format(precision_score(vali_epoch_gt, vali_epoch_preds, average='macro')),
              # "Vali Rcll: {:.4f}".format(recall_score(vali_epoch_gt, vali_epoch_preds, average='macro')),
               "Vali F1: {:.4f}".format(f1_score(vali_epoch_gt, vali_epoch_preds, average=None)[-1]))
              # "\nTest Loss: {:.4f}".format(np.mean(test_epoch_losses)),
              # "Test Acc: {:.4f}".format(accuracy_score(test_epoch_gt, test_epoch_preds)),
              # "Test F1: {:.4f}".format(f1_score(test_epoch_gt, test_epoch_preds, average=None)[-1]))
        
        # save_best_model(np.mean(vali_epoch_losses), e, model, optimizer, criterion)
        # print('-'*50)
        if config['early_stopping']:
            early_stopping(np.mean(vali_epoch_losses), model) 
            # early_stopping(f1_score(vali_epoch_gt, vali_epoch_preds, average='macro')*-1, model) 
            if early_stopping.early_stop:
                print("Early stopping")
                break

                
        
    # load the last checkpoint with the best model
    if config['early_stopping']:
        model.load_state_dict(torch.load(f"checkpoint.pt"))
        # model.load_state_dict(torch.load(config['output_path'] +f"checkpoint_retrain_{config['c_subject']}.pt"))

    
    plot_train_vali_nestedCV(loss_train, loss_vali, acc_train, acc_vali, f1_train, f1_vali, config)
    
    
    #training
    train_epoch_losses, train_epoch_preds,train_epoch_gt, train_epoch_output = validate(model, trainloader, criterion)
    # confu_train = confusion_matrix(train_epoch_gt, train_epoch_preds)
    # confu_df = pd.DataFrame(confu_train, index=['Non-Walking', 'Walking', 'Ramp', 'Staris'], columns=['Non-Walking', 'Walking', 'Ramp', 'Staris'])
    Acc= accuracy_score(train_epoch_gt, train_epoch_preds)
    Sen= recall_score(train_epoch_gt, train_epoch_preds, average=None, labels=np.unique(train_epoch_gt))[-1]
    Pre= precision_score(train_epoch_gt, train_epoch_preds, average=None, labels=np.unique(train_epoch_gt))[-1]
    f1= f1_score(train_epoch_gt, train_epoch_preds, average=None, labels=np.unique(train_epoch_gt))[-1]
    roc = roc_auc_score(train_epoch_gt, train_epoch_output, average=None, labels=np.unique(train_epoch_gt))
    Performance.iloc[0,0:5]=np.array([Acc, Sen, Pre, f1, roc], dtype=np.float32)  
    
    #validation
    vali_epoch_losses, vali_epoch_preds,vali_epoch_gt, vali_epoch_output  = validate(model, valiloader, criterion)
    # confu_vali = confusion_matrix(vali_epoch_gt, vali_epoch_preds)
    # confu_df = pd.DataFrame(confu_vali, index=['Non-Walking', 'Walking', 'Ramp', 'Staris'], columns=['Non-Walking', 'Walking', 'Ramp', 'Staris'])
    Acc= accuracy_score(vali_epoch_gt, vali_epoch_preds)
    Sen= recall_score(vali_epoch_gt, vali_epoch_preds, average=None, labels=np.unique(vali_epoch_gt))[-1]
    Pre= precision_score(vali_epoch_gt, vali_epoch_preds, average=None, labels=np.unique(vali_epoch_gt))[-1]
    f1= f1_score(vali_epoch_gt, vali_epoch_preds, average=None, labels=np.unique(vali_epoch_gt))[-1]
    roc = roc_auc_score(vali_epoch_gt, vali_epoch_output, average=None, labels=np.unique(vali_epoch_gt))
    Performance.iloc[0,5:10]=np.array([Acc, Sen, Pre, f1, roc], dtype=np.float32)  
    
    #Average Performance
    Performance_ave = pd.DataFrame(Performance.mean()[1:]).T
    test_epoch_preds,test_epoch_gt, test_epoch_output, WalkTime = test_window(valiloader, model, criterion, config)
    
    return  Performance_ave, WalkTime



