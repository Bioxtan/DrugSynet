import os
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn.metrics as metrics
from dataset import SynergyEncoderDataset
#from network import BERTCSDTAmodel, BERTDTAmodel, net_reg, MuSigma, BERTGMMDTAmodel
from sklearn.model_selection import KFold, ShuffleSplit,train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from dgllife.utils import EarlyStopping
import argparse
import os
from sklearn.metrics import accuracy_score,classification_report,precision_recall_curve,average_precision_score,roc_auc_score#改成分类
from imblearn.metrics import sensitivity_score,specificity_score
from sklearn.metrics import balanced_accuracy_score,cohen_kappa_score
from sklearn.metrics import matthews_corrcoef,confusion_matrix
from sklearn.metrics import f1_score,recall_score,precision_score
import torch.nn.functional as F
from smiles_to_graph import smiles_to_graph
from prettytable import PrettyTable
from typing import ClassVar, Iterable, Mapping, Optional, Sequence, Tuple, Union
from collections import defaultdict
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
#修改线
from model import MultiViewNet
import json
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from ipdb import set_trace


def compute_kl_loss(p, q, pad_mask=None):
	p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
	q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

	# pad_mask is for seq-level tasks
	if pad_mask is not None:
		p_loss.masked_fill_(pad_mask, 0.)
		q_loss.masked_fill_(pad_mask, 0.)

	# You can choose whether to use function "sum" and "mean" depending on your task
	p_loss = p_loss.mean()
	q_loss = q_loss.mean()

	loss = (p_loss + q_loss) / 2
	return loss
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def read_data_file(data_file):
    smiles_1 = []
    smiles_2 = []
    Y = []
    context=[]

    with open(data_file, 'r') as f:
        all_lines = f.readlines()

        for line in all_lines[1:]:
            row = line.rstrip().split(',')[0:]
            # print(row)
            smiles_1.append(row[0])
            smiles_2.append(row[1])
            context.append((row[2]))
            Y.append(int(row[3]))

    return smiles_1, smiles_2,context,Y


def define_dataloader(train_index,test_index,smiles_1,smiles_2,context,Y,maxLen,batch_size):
    test_index, valid_index = train_test_split(test_index, test_size=0.5, random_state=42)
    train_drug_2_cv = np.array(smiles_2)[train_index]
    train_drug_1_cv = np.array(smiles_1)[train_index]
    train_context_cv = np.array(context)[train_index]
    train_Y_cv = np.array(Y)[train_index]
    
    test_drug_2_cv = np.array(smiles_2)[test_index]
    test_drug_1_cv = np.array(smiles_1)[test_index]
    test_context_cv = np.array(context)[test_index]
    test_Y_cv = np.array(Y)[test_index]

    valid_drug_2_cv = np.array(smiles_2)[valid_index]
    valid_drug_1_cv = np.array(smiles_1)[valid_index]
    valid_context_cv = np.array(context)[valid_index]
    valid_Y_cv = np.array(Y)[valid_index]

    train_dataset = SynergyEncoderDataset(train_drug_1_cv,
                                           train_drug_2_cv,
                                           train_Y_cv,
                                           train_context_cv
                                          ,maxLen, device=device)
    
    valid_dataset = SynergyEncoderDataset(valid_drug_1_cv,
                                           valid_drug_2_cv,
                                           valid_Y_cv,
                                           valid_context_cv
                                          ,maxLen, device=device)

    test_dataset = SynergyEncoderDataset( test_drug_1_cv,
                                          test_drug_2_cv,
                                          test_Y_cv,
                                          test_context_cv
                                         ,maxLen, device=device)
    trainLoader = DataLoader( train_dataset,
            batch_size= batch_size, shuffle=True )
    
    validLoader = DataLoader( valid_dataset,
            batch_size= batch_size, shuffle=True )
    
    testLoader = DataLoader( test_dataset,
            batch_size= batch_size, shuffle=False )

    return trainLoader, validLoader, testLoader

def train(train_loader, model, epoch, optimizer, device, scheduler, print_freq=200):
    model.train()
    cross_entropy_loss = nn.CrossEntropyLoss()
    losses = AverageMeter()

    for i, batch in enumerate(train_loader):#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        optimizer.zero_grad()
        drug1_graph,drug2_graph,drug_info= batch
        drug1_graph,drug2_graph = drug1_graph.to(device),drug2_graph.to(device)
        compounds_1, compounds_2,synergyScores, context,fp1,fp2 = (drug_info[0]).to(device), (drug_info[1]).to(device),(drug_info[5]).to(device), (drug_info[4]).to(device),(drug_info[2]).to(device),(drug_info[3]).to(device)
        
        
        pre_synergy = model(drug1_graph,drug2_graph,compounds_1,compounds_2,context,fp1,fp2)

        pre_synergy2 = model(drug1_graph,drug2_graph,compounds_1,compounds_2,context,fp1,fp2)

        ce_loss = 0.5 * (cross_entropy_loss(pre_synergy, synergyScores.squeeze(1)) + cross_entropy_loss(pre_synergy2, synergyScores.squeeze(1)))
        kl_loss = compute_kl_loss(pre_synergy, pre_synergy2)
        α = 5
        loss = ce_loss + α * kl_loss
 
        losses.update( loss.item(), len(compounds_1))
        loss.backward()
        scheduler.step()
        if np.isnan( loss.item() ):#以元素方式测试NaN，并以布尔数组的形式返回结果
            raise Exception("Training model diverges.")
        optimizer.step()

        if i % print_freq == 0:
            log_str = 'TRAIN -> Epoch{epoch}: \tIter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f})'.format( epoch=epoch, iter=i , loss=losses )
            print( log_str )

def validate_new(valid_loader, model):#valid_loader
    model.eval()#设为评估模式
    preds = torch.Tensor()
    trues = torch.Tensor()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            drug1_graph,drug2_graph,drug_info= batch
            drug1_graph,drug2_graph = drug1_graph.to(device),drug2_graph.to(device)            
            compounds_1, compounds_2,Y, context,fp1,fp2 = (drug_info[0]).to(device), (drug_info[1]).to(device),(drug_info[5]).to(device), (drug_info[4]).to(device),(drug_info[2]).to(device),(drug_info[3]).to(device)
            
            pre_synergy = model(drug1_graph,drug2_graph,compounds_1, compounds_2,context,fp1,fp2)#squeeze内置0，synergy.todevice改成y，不知道为什么先试试
            pre_synergy = torch.nn.functional.softmax(pre_synergy)[:,1]
            preds = torch.cat((preds, pre_synergy.cpu()), 0)#cat:多个tensor拼接。删除pre_synergy.argmax(-1)，使其一维直接输出
            trues = torch.cat((trues, Y.view(-1, 1).cpu()), 0)

        y_pred = np.array(preds) > 0.5#np.where(preds>0.5,1,0)
        accuracy = accuracy_score(trues, y_pred)
        BACC = balanced_accuracy_score(trues, y_pred)
        roc_auc = roc_auc_score(trues, preds)
        ACC = accuracy_score(trues, y_pred)
        F1 = f1_score(trues, y_pred, average='binary')
        Prec = precision_score(trues, y_pred, average='binary')
        Rec = recall_score(trues, y_pred, average='binary')
        kappa = cohen_kappa_score(trues, y_pred)
        mcc = matthews_corrcoef(trues, y_pred)
        ap = average_precision_score(trues, preds)
        # Confuse  = confusion_matrix(trues,y_pred)

        # file2_name = 'preds.txt'
        # with open(file2_name, 'w') as fin:
        #     fin.write('true,pred\n')
        #     t = np.array(trues)
        #     t_1 = list(t)
        #     t_lat = ",".join(str(x) for x in t_1)
        #     y1 = list(y_pred)
        #     y_lat = ','.join(str(x) for x in y1)
        #     for i, batch in enumerate(valid_loader):
        #         fin.write(t_lat + ',' + y_lat )#打印出预测值和真实值

        return accuracy,ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap
    
def setup_seed(modelseed=23):
    random.seed(modelseed)
    np.random.seed(modelseed)
    torch.manual_seed(modelseed)
    torch.cuda.manual_seed(modelseed)
    torch.cuda.manual_seed_all(modelseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True #Set for reproducibility

if __name__ == "__main__":
    
    setup_seed(42)

    device = torch.device("cuda:1") 
    all_loss =  np.zeros((5,1))
    all_accuracy = np.zeros((5,1))
    all_report = np.zeros((5,1))
    all_ACC = np.zeros((5,1))
    all_BACC = np.zeros((5,1))
    all_Prec = np.zeros((5,1))
    all_Rec  = np.zeros((5,1))
    all_F1   = np.zeros((5,1))
    all_roc_auc = np.zeros((5,1))
    all_mcc = np.zeros((5,1))
    all_kappa = np.zeros((5,1))
    all_ap = np.zeros((5,1))
    n_epochs = 500
   
    smiles_1, smiles_2,context,Y = read_data_file('/home/th_xy/SynPred/triple.csv')#FLAGS.dataset_path)
    
    kf = KFold(n_splits=5, shuffle=True)#划分成8:1 之前是5，

    for split, (train_index, test_index) in enumerate( kf.split(Y)):
        trainLoader, validLoader, testLoader = define_dataloader( train_index, test_index,smiles_1, smiles_2,context, Y,
                                                                  128,  256)#batchsize，=128 改成512 

        model = MultiViewNet()#写在外部导致k折验证的valid会已经被验证过，而写在内部则会更新model，保证验证集不是已经学过的。
        model.to(device)
        stopper = EarlyStopping(mode='higher', filename='mainsplit-attention-comb',patience=30)
        model = model.to(device)
        lr = 1e-4 #0.00005
        optimizer = AdamW(model.parameters(), lr=lr)#初始lr=1e-3
        scheduler =  torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=n_epochs,
                                          steps_per_epoch=len(trainLoader))
        for epochind in range(n_epochs):
            train(trainLoader, model, epochind, optimizer,device,scheduler)
            accuracy,ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap = validate_new(validLoader, model)
            e_tables = PrettyTable(['epoch', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC',  'kappa', 'ap'])
            e_tables.float_format = '.3' 
            row = [epochind,ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap]
            e_tables.add_row(row)
            print(e_tables)
            early_stop = stopper.step(ACC, model)
            if early_stop:
                break
        stopper.load_checkpoint(model)
        accuracy, ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap= validate_new(testLoader, model)

        e_tables = PrettyTable(['test', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC',  'kappa', 'ap'])
        e_tables.float_format = '.3' 
        row = [epochind,ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap]
        e_tables.add_row(row)
        print(e_tables)
        all_accuracy[split] = accuracy
        all_ACC[split] = ACC
        all_BACC[split] = BACC
        all_Prec[split] = Prec
        all_Rec[split] = Rec
        all_F1[split] = F1
        all_roc_auc[split] = roc_auc
        all_mcc[split] = mcc
        all_kappa[split] = kappa
        all_ap[split] =ap


    print('*='*20)
    print('accuracy:  {0:6f}({1:6f})'.format(np.mean(all_accuracy),  np.std(all_accuracy)))
    print('ACC:  {0:6f}({1:6f})'.format(np.mean(all_ACC), np.std(all_ACC)))
    print('BACC:  {0:6f}({1:6f})'.format(np.mean(all_BACC), np.std(all_BACC)))
    print('Prec:  {0:6f}({1:6f})'.format(np.mean(all_Prec), np.std(all_Prec)))
    print('Rec:  {0:6f}({1:6f})'.format(np.mean(all_Rec), np.std(all_Rec)))
    print('F1:  {0:6f}({1:6f})'.format(np.mean(all_F1), np.std(all_F1)))
    print('roc_auc:  {0:6f}({1:6f})'.format(np.mean(all_roc_auc), np.std(all_roc_auc)))
    print('mcc:  {0:6f}({1:6f})'.format(np.mean(all_mcc), np.std(all_mcc)))
    print('kappa:  {0:6f}({1:6f})'.format(np.mean(all_kappa), np.std(all_kappa)))
    print('ap:  {0:6f}({1:6f})'.format(np.mean(all_ap), np.std(all_ap)))
    # print('confuse:'.format(confuse))#增加混淆矩阵分析