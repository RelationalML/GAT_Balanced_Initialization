from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures,LargestConnectedComponents
from torch_geometric.utils import segregate_self_loops
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv
import numpy as np
import matplotlib.pyplot as plt
import copy
import json
import pprint
import pickle
from scipy import stats
import pandas as pd
import os.path
from math import floor,ceil
import torch_geometric.transforms as T
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

path = "hetExp/ExpResultsHetGATv2/"
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getData(datasetName,dataTransform):
    dataset = Planetoid(root='data/Planetoid', name=datasetName, transform=NormalizeFeatures())
    data = dataset[0]
    if dataTransform=='removeIsolatedNodes':
        out = segregate_self_loops(data.edge_index)
        edge_index, edge_attr, loop_edge_index, loop_edge_attr = out
        mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
        mask[edge_index.view(-1)] = 1
        data.train_mask = data.train_mask & mask
        data.val_mask = data.val_mask & mask
        data.test_mask = data.test_mask & mask
    if dataTransform=='useLCC':
        transformLCC = LargestConnectedComponents()
        data = transformLCC(data)
    return data,dataset.num_features,dataset.num_classes


def getDataHet(datasetName):
    print("Loading datasets as npz-file..")
    data = np.load('hetExp/data/'+datasetName+'.npz')
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    num_targets = 1 if num_classes == 2 else num_classes
    
    print("Converting to PyG dataset...")
    data = Data(x=x, edge_index=edge_index)
    data.y = y
    data.num_classes = num_classes
    data.num_targets = num_targets
    data.train_mask = train_mask[:,1] #split_idx = 1, 10 splits provided in dataset
    data.val_mask = val_mask[:,1]
    data.test_mask = test_mask[:,1]
    return data,data.num_features,data.num_classes

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class GATv2(torch.nn.Module):
    def __init__(self, numLayers, dims, heads, concat, weightSharing, attnDropout=0,bias=False,activation='relu'):
        super().__init__()
        self.numLayers = numLayers
        self.heads = heads
        self.weightSharing = weightSharing
        self.dropout = attnDropout
        if activation=='relu':
            self.activation = F.relu
        elif activation=='elu':
            self.activation = F.elu # as used previously
        self.GATv2Convs = torch.nn.ModuleList(
            [GATv2Conv(dims[j]*heads[j],dims[j+1],bias=bias,
                       heads=heads[j+1],concat=concat[j],share_weights=weightSharing,dropout=attnDropout) 
                       for j in range(self.numLayers)])

        for i in range(len(self.GATv2Convs)):
            # weights go through relu in neighborhood aggregation, but leakyrelu in attention coefficient calculation
            torch.nn.init.xavier_normal_(self.GATv2Convs[i].lin_l.weight.data)#,gain=torch.nn.init.calculate_gain('relu'))
            if not self.weightSharing:
                torch.nn.init.xavier_normal_(self.GATv2Convs[i].lin_r.weight.data)#gain=torch.nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_normal_(self.GATv2Convs[i].att.data)#,gain=torch.nn.init.calculate_gain('relu'))
         
    def forward(self, x, edge_index,getAttnCoef):
         #leakyrelu for computing alphas have negative_slope=0.2 (as set in GAT and used in GATv2)
        attnCoef = [0] * len(self.GATv2Convs)
        for i in range(len(self.GATv2Convs)-1):
            x,a = self.GATv2Convs[i](x,edge_index,return_attention_weights=getAttnCoef)
            attnCoef[i] = (a[0].detach(),a[1].detach())
            x = self.activation(x)#x.relu() #F.relu(x,inplace=True) 
            if self.dropout>0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x,a = self.GATv2Convs[len(self.GATv2Convs)-1](x,edge_index,return_attention_weights=getAttnCoef)
        attnCoef[len(self.GATv2Convs)-1] =  (a[0].detach(),a[1].detach())
        return x,attnCoef

def computeStatSumry(arr,quantiles):
  r = {'mean': arr.mean(),
        'std': arr.std()}
  quantiles=torch.cat((torch.tensor([0,1],device=device),quantiles),dim=0)
  p = torch.quantile(arr,quantiles)
  r['min'] = p[0]
  r['max'] = p[1]
  for i in range(2,len(quantiles)):
    r[str(int(quantiles[i]*100))+'%ile'] = p[i]
  return r

def computeAlphaStatSumry(alphas,quantiles):
    return [computeStatSumry(alphas[1][np.where(np.equal(alphas[0][0],alphas[0][1])==True)[0]],quantiles),
       computeStatSumry(alphas[1][np.where(np.equal(alphas[0][0],alphas[0][1])==False)[0]],quantiles)]

def makeDataDimsEven(data,input_dim,output_dim):
    if input_dim%2==1:
        a=torch.zeros((data.x.size()[0],ceil(data.x.size()[1]/2)*2))
        a[:,:input_dim] = data.x
        data.x = a
        input_dim+=1
    output_dim=(ceil(output_dim/2))*2
    return data,input_dim,output_dim

def printExpSettings(expID,expSetting):
    print('Exp: '+str(expID))
    for k,v in expSetting.items():
        for k2,v2 in expSetting[k].items():
            if(k2==expID):
                print(k,': ',v2)

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initializeParams(params,initScheme,activation):#'xavierN','xavierU','kaimingN','kaimingU','LLxavierN','LLxavierU',LLkaimingN','LLkaimingU','LLortho'
    # if activation=='relu':
    #     gain=torch.sqrt(2)
    # elif activation=='elu':
    #     gain=1 #or 3/4 for SELU?
    numLayers = len(params)
    paramTypes = params[0].keys()
    with torch.no_grad():
        if(initScheme[:2]!='LL'):
            for l in range(numLayers):
                for f in set(paramTypes):
                    if(initScheme=='xavierN'):
                        torch.nn.init.xavier_normal_(params[l][f].data)
                    if(initScheme=='xavierU'):
                        torch.nn.init.xavier_uniform_(params[l][f].data)
                    if(initScheme=='kaimingN'):
                        torch.nn.init.kaiming_normal_(params[l][f].data,mode='fan_in',nonlinearity=activation)
                    if(initScheme=='kaimingU'):
                        torch.nn.init.kaiming_uniform_(params[l][f].data,mode='fan_in',nonlinearity=activation)
        elif(initScheme[:2]=='LL'):
            for l in range(numLayers):
                params[l]['attn'].data = torch.zeros(params[l]['attn'].data.shape,device=device) ##LL attnWeights are 0
            for f in set(paramTypes)-set(['attn']):
                firstLayerDeltaDim = (ceil(params[0][f].data.shape[0]/2),params[0][f].data.shape[1])
                finalLayerDeltaDim= (params[numLayers-1][f].data.shape[0],ceil(params[numLayers-1][f].data.shape[1]/2))
                if initScheme=='LLxavierU':
                    firstLayerDelta = torch.nn.init.xavier_uniform_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device))
                    finalLayerDelta = torch.nn.init.xavier_uniform_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device))
                if initScheme=='LLxavierN':
                    firstLayerDelta = torch.nn.init.xavier_normal_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device))
                    finalLayerDelta = torch.nn.init.xavier_normal_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device))
                if initScheme=='LLkaimingU':
                    firstLayerDelta = torch.nn.init.kaiming_uniform_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device),nonlinearity=activation)
                    finalLayerDelta = torch.nn.init.kaiming_uniform_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device),nonlinearity=activation)
                if initScheme=='LLkaimingN':
                    firstLayerDelta = torch.nn.init.kaiming_normal_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device),nonlinearity=activation)
                    finalLayerDelta = torch.nn.init.kaiming_normal_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device),nonlinearity=activation)
                if initScheme=='LLortho':
                    firstLayerDelta = torch.nn.init.orthogonal_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=device))
                    finalLayerDelta = torch.nn.init.orthogonal_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=device))
                params[0][f].data = torch.cat((firstLayerDelta,-firstLayerDelta),dim=0) #BUG CHECK
                params[numLayers-1][f].data = torch.cat((finalLayerDelta,-finalLayerDelta),dim=1) #BUG CHECK
            for l in range(1,numLayers-1):
                    for f in set(paramTypes)-set(['attn']):
                        dim = params[l][f].data.shape
                        if initScheme=='LLxavierU':
                            delta = torch.nn.init.xavier_uniform_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device))
                        if initScheme=='LLxavierN':
                            delta = torch.nn.init.xavier_normal_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device))
                        if initScheme=='LLkaimingU':
                            delta = torch.nn.init.kaiming_uniform_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device),nonlinearity=activation)
                        if initScheme=='LLkaimingN':
                            delta = torch.nn.init.kaiming_normal_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device),nonlinearity=activation)
                        if initScheme=='LLortho':
                            delta = torch.nn.init.orthogonal_(torch.empty(ceil(dim[0]/2),ceil(dim[1]/2),device=device))
                        delta = torch.cat((delta, -delta), dim=0)
                        delta = torch.cat((delta, -delta), dim=1)
                        params[l][f].data = delta
        if(initScheme=='xavrWzeroA'):
            for l in range(numLayers):
                torch.nn.init.zeros_(params[l]['attn'].data)
                for f in set(paramTypes)-set(['attn']):
                    torch.nn.init.xavier_normal_(params[l][f].data)
    for l in range(numLayers):
        for f in paramTypes:
            params[l][f].data.requires_grad=True #because of initialization update
    return params

def scaleParams(params,scalScheme,scalHP):#'balLtoRconst','balLtoRuniform','balLtoRnormal','balRtoLconst','balRtoLuniform','balRtoLnormal'
    numLayers = len(params)
    paramTypes = params[0].keys()
    beta = float(scalHP[2])
    with torch.no_grad():
        if scalScheme in ['balLtoRconst','balLtoRuniform','balLtoRnormal']:
            for f in set(paramTypes)-set(['attn']):
                incSqNorm = torch.sqrt(torch.pow(params[0][f].data,2).sum(axis=1))
                if scalScheme=='balLtoRuniform':
                    reqRowWiseSqL2Norm = torch.randint(low=int(scalHP[0]),high=int(scalHP[1]),size=(params[0][f].data.size()[0],),device=device)
                if scalScheme=='balLtoRnormal':
                    reqRowWiseSqL2Norm = float(scalHP[0]) + float(scalHP[1])*(torch.randn((params[0][f].data.size()[0],),device=device))
                if scalScheme=='balLtoRconst':
                    reqRowWiseSqL2Norm = torch.full((params[0][f].data.size()[0],),float(scalHP[0]),device=device)
                params[0][f].data = torch.multiply(torch.divide(params[0][f].data,incSqNorm.reshape((len(incSqNorm),1))),\
                    torch.sqrt(reqRowWiseSqL2Norm.reshape(len(reqRowWiseSqL2Norm),1)))
            for l in range(1,numLayers):
                attnSqNormReq = 0
                for f in set(paramTypes)-set(['attn']):
                    incSqNorm = torch.pow(params[l-1][f].data,2).sum(axis=1)
                    outSqNorm = torch.sqrt(torch.pow(params[l][f].data,2).sum(axis=0))
                    params[l][f].data = torch.multiply(torch.divide(params[l][f].data,outSqNorm.reshape((1,len(outSqNorm)))),\
                                            torch.sqrt((incSqNorm*beta).reshape((1,len(incSqNorm)))))#torch.sqrt(min(incSqNorm))#
                    outSqNorm = torch.pow(params[l][f].data,2).sum(axis=0)#*torch.sqrt(min(incSqNorm))
                    attnSqNormReq += incSqNorm-outSqNorm
                if beta==1: #beta=1 -> attnWeghts should be 0 for balanced scaling 
                    params[l-1]['attn'].data = torch.zeros(params[l-1]['attn'].data.shape,device=device)
                else:
                    params[l-1]['attn'].data = torch.sqrt(attnSqNormReq).reshape(params[l-1]['attn'].data.shape) #to genarlize for multiple heads by reshaping the sqnormreq vector
            #set attn for layer l? set with zeros for now (next line) - rethink
            params[numLayers-1]['attn'].data = torch.zeros(params[numLayers-1]['attn'].data.shape,device=device)
    
        if scalScheme in ['balRtoLconst','balRtoLuniform','balRtoLnormal']:
                for f in set(paramTypes)-set(['attn']):
                    outSqNorm = torch.sqrt(torch.pow(params[numLayers-1][f].data,2).sum(axis=0))
                    if initScheme=='balRtoLuniform':
                        reqColWiseSqL2Norm = torch.randint(low=int(scalHP[0]),high=int(scalHP[1]),size=(params[numLayers-1][f].data.size()[1],),device=device)
                    if initScheme=='balRtoLnormal':
                        reqColWiseSqL2Norm = float(scalHP[0]) + float(scalHP[1])*(torch.randn((params[numLayers-1][f].data.size()[1],),device=device))
                    if initScheme=='balRtoLconst':
                        reqColWiseSqL2Norm = torch.full((params[numLayers-1][f].data.size()[1],),float(scalHP[0]),device=device)
                    params[numLayers-1][f].data = torch.multiply(torch.divide(params[numLayers-1][f].data,outSqNorm.reshape((1,len(outSqNorm)))),\
                                                torch.sqrt(reqColWiseSqL2Norm.reshape(1,len(reqColWiseSqL2Norm))))
                for l in range(numLayers-2,-1,-1):
                    attnSqNormReq = 0
                    for f in set(paramTypes)-set(['attn']):
                        outSqNorm = torch.pow(params[l+1][f].data,2).sum(axis=0)
                        incSqNorm = torch.sqrt(torch.pow(params[l][f].data,2).sum(axis=1))
                        params[l][f].data = torch.divide(params[l][f].data,incSqNorm.reshape((len(incSqNorm),1)))\
                                                *torch.sqrt((outSqNorm*beta).reshape((len(outSqNorm),1)))#torch.sqrt(min(incSqNorm))#
                        incSqNorm = torch.pow(params[l][f].data,2).sum(axis=1)#*torch.sqrt(min(incSqNorm))
                        attnSqNormReq += incSqNorm-outSqNorm
                    if beta==1: #beta=1 -> attnWeghts should be 0 for balanced scaling 
                        params[l]['attn'].data = torch.zeros(params[l-1]['attn'].data.shape,device=device)
                    else:
                        params[l]['attn'].data = torch.sqrt(attnSqNormReq).reshape(params[l]['attn'].data.shape) #to genarlize for multiple heads by reshaping the sqnormreq vector
                #set attn for layer l? set with zeros for now (next line) - rethink
                params[numLayers-1]['attn'].data = torch.zeros(params[numLayers-1]['attn'].data.shape,device=device)
    
    for l in range(numLayers):
        for f in paramTypes:
            params[l][f].data.requires_grad=True
    return params

def deepCopyParamsToNumpy(params):
    paramsCopy = [{} for i in range(len(params))]    
    for l in range(len(params)):
        for p in params[l].keys():
            paramsCopy[l][p] = params[l][p].data.detach().cpu().numpy()
    return paramsCopy

expSetting = pd.read_csv('hetExp/initExpSettingsGATv2Het.csv',index_col='expId').fillna('').to_dict()
expIDs =  range(178,186+1) #72,79, 80,87
runIDs =[]# [0,1]#1]
trainLossToConverge = 0.001
printLossEveryXEpoch = 500
saveParamGradStatSumry = False
saveNeuronLevelL2Norms = False
saveLayerWiseForbNorms = False
saveWeightsAtMaxValAcc = False
quantiles = torch.tensor((np.array(range(1,10,1))/10),dtype=torch.float32,device=device)
qLabels = [str(int(q*100))+'%ile' for q in quantiles]
labels = ['min','max','mean','std']+qLabels 

for expID in expIDs:
    numRuns = int(expSetting['numRuns'][expID])
    if len(runIDs)==0:
        runIDs = range(numRuns) #or specify
    datasetName = str(expSetting['dataset'][expID])
    optim = str(expSetting['optimizer'][expID])
    numLayers = int(expSetting['numLayers'][expID])
    numEpochs = int(expSetting['maxEpochs'][expID])
    lr = float(expSetting['initialLR'][expID])
    if int(expSetting['hiddenDim'][expID]) < 0:
        if int(expSetting['hiddenDim'][expID])==-5:
            hiddenDims = [128,64,32,16]
        if int(expSetting['hiddenDim'][expID])==-10:
            hiddenDims = [256,128,128,64,64,32,32,16,16]
            #hiddenDims = [384,256,192,128,96,64,48,32,24]
        print('HiddenDims: ',hiddenDims)
    else:
        hiddenDims = [int(expSetting['hiddenDim'][expID])] * (numLayers-1)
    heads = [1] + ([int(expSetting['attnHeads'][expID])] * (numLayers-1)) + [1] #last layer and  datainput always have 1 attention head
    concat = ([True] * (numLayers-1)) + [False] #concat attn heads for all layers except the last, avergae for last (doesn't matter when num of heads for last layer=1)
    attnDropout = float(expSetting['attnDropout'][expID])
    wghtDecay =  float(expSetting['wghtDecay'][expID])
    activation = str(expSetting['activation'][expID])
    weightSharing = bool(expSetting['weightSharing'][expID])
    dataTransform = str(expSetting['dataTransform'][expID]) #removeIsolatedNodes,useLCC #note: these transforms may change the number of train/val/test nodes
    initScheme=str(expSetting['initScheme'][expID])
    scalScheme=str(expSetting['scalScheme'][expID])
    lrDecayFactor = float(expSetting['lrDecayFactor'][expID])
    if lrDecayFactor<1:
        lrDecayPatience = float(expSetting['lrDecayPatience'][expID])
    scalHPstr = [0,0,0]
    if len(str(expSetting['scalHP'][expID]))>0:
         scalHPstr=[float(x) for x in str(expSetting['scalHP'][expID]).split('|')] #e.g. (low,high) for uniform, (mean,std) for normal, (const) for const
 

    data,input_dim,output_dim = getDataHet(datasetName)#getData(datasetName,dataTransform) 
    data = data.to(device)
    dims = [input_dim]+hiddenDims+[output_dim]
    if weightSharing:
        paramTypes = ['feat','attn']
    else:
        paramTypes = ['feat','feat2','attn']
    recordAlphas = False
    print('*******')
    printExpSettings(expID,expSetting)
    
    for run in runIDs:#range(numRuns):
        print('-- RUN ID: '+str(run))
        set_seeds(run)
        model = GATv2(numLayers,dims,heads,concat, weightSharing,attnDropout,activation=activation).to(device)
        if optim=='SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wghtDecay)
        if optim=='Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wghtDecay)
        criterion = torch.nn.CrossEntropyLoss()
        if lrDecayFactor<1:
            lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=lrDecayFactor, patience=lrDecayPatience) #based on valAcc
        
        trainLoss = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        valLoss = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        trainAcc = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        valAcc = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        testAcc = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        
        # #extra records of parameters for studying training dynamics
        if saveParamGradStatSumry:
            paramStatSumry = [{} for i in range(numLayers)]
            for i in range(numLayers):
                for f in paramTypes:
                    paramStatSumry[i][f] = {x2:{x:torch.zeros(numEpochs,device=device) for x in labels} for x2 in ['wght','grad']}
        if saveNeuronLevelL2Norms:
            featL2Norms = [{} for i in range(numLayers)]
            attnWghtsSq = [torch.zeros((numEpochs,dims[i+1]),device=device) for i in range(numLayers)]
            for i in range(numLayers):
                for f in set(paramTypes)-set(['attn']): #incoming: row-wise of W matrix, and outgoing is col-wise of W matrix
                    featL2Norms[i][f] =  {'row':torch.zeros((numEpochs,dims[i+1]),device=device),'col':torch.zeros((numEpochs,dims[i]),device=device)}
        if saveLayerWiseForbNorms:
            forbNorms = [{f:{x:torch.zeros(numEpochs, device=device) for x in ['wght','grad']}
                             for f in paramTypes} for i in range(numLayers)]

        # #changeInParamStatSumry = [{} for i in range(numLayers)]
        # #prevRec = [{f:{'wght':None} for f in paramTypes} for i in range(numLayers)]
        # #currRec = [{f:{'wght':None,'grad':None} for f in paramTypes} for i in range(numLayers)]
        # #alphaStatSumry = [{x2:{x:np.zeros(numEpochs) for x in labels} for x2 in ['alpha_ii','alpha_ij']} for i in range(numLayers)] 
        # for i in range(numLayers):
        #     for f in paramTypes:
        #         #changeInParamStatSumry[i][f] = {'wght':{x:np.zeros(numEpochs) for x in labels}} 
        #        
        
        #map default param names to custom names to match visualization scripts later
        modelParamNameMapping = {'att':'attn','lin_l':'feat','lin_r':'feat2'}
        params = [{} for i in range(numLayers)]
        for name,param in model.named_parameters():
            paramNameTokens = name.split('.')
            params[int(paramNameTokens[1])][modelParamNameMapping[paramNameTokens[2]]] = param

        params = initializeParams(params,initScheme,activation)
        params = scaleParams(params,scalScheme,scalHPstr)
        paramsAtMaxValAcc = None

        initialParamsCopy  = deepCopyParamsToNumpy(params)

        maxValAcc = 0
        continueTraining = True      
        epoch=0
        while(epoch<numEpochs and continueTraining): 
            
            
            #record required quantities of weights used in a layer
            if saveParamGradStatSumry:
                for l in range(numLayers):
                    for p in paramTypes:
                        for k,v in computeStatSumry(params[l][p].data.detach(),quantiles).items():
                            paramStatSumry[l][p]['wght'][k][epoch] = v
            if saveNeuronLevelL2Norms:
                for l in range(numLayers):
                    for p in paramTypes:
                        wghts=params[l][p].data.detach()
                        if p=='attn':
                            attnWghtsSq[l][epoch] = torch.pow(wghts,2)
                        else:
                            featL2Norms[l][p]['row'][epoch] = torch.pow(wghts,2).sum(axis=1)
                            featL2Norms[l][p]['col'][epoch] = torch.pow(wghts,2).sum(axis=0)
            if saveLayerWiseForbNorms:
                for l in range(numLayers):
                    for p in paramTypes:
                        forbNorms[l][p]['wght'][epoch] = torch.sqrt(torch.pow(params[l][p].data.detach(),2).sum())
            # for l in range(numLayers):
            #     for p in paramTypes:
            #         #prevRec[l][p]['wght'] = params[l][p].data.detach().cpu().numpy()
            #         if p=='attn':
            #             #prevRec[l][p]['wght'] = prevRec[l][p]['wght'][0]
            
            model.train()
            optimizer.zero_grad()  # Clear gradients.

            out,attnCoef = model(data.x, data.edge_index,getAttnCoef=recordAlphas)  # Perform a single forward pass.
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
            trainLoss[epoch] = loss.detach()
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            train_correct = pred[data.train_mask] == data.y[data.train_mask]  # Check against ground-truth labels.
            trainAcc[epoch] = int(train_correct.sum()) / int(data.train_mask.sum())  # Derive ratio of correct predictions.
            #trainAcc[epoch] = roc_auc_score(data.y[data.train_mask].cpu(),pred[data.train_mask].cpu())
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

            # for l in range(numLayers):
            #     for p in paramTypes:
            #         print(l,' ',p,' ',params[l][p].grad)

            #record quantities again for the gradients in the epoch 
            if saveParamGradStatSumry:
                for l in range(numLayers):
                    for p in paramTypes:
                        for k,v in computeStatSumry(params[l][p].grad.detach(),quantiles).items():
                            paramStatSumry[l][p]['grad'][k][epoch] = v
            if saveLayerWiseForbNorms:
                for l in range(numLayers):
                    for p in set(paramTypes):
                        forbNorms[l][p]['grad'][epoch] = torch.sqrt(torch.pow(params[l][p].grad.detach(),2).sum())

            model.eval()
            with torch.no_grad():
                out,a = model(data.x, data.edge_index,getAttnCoef=False)
                valLoss[epoch] = criterion(out[data.val_mask], data.y[data.val_mask]).detach() #getValLoss().data.detach()
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                val_correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
                valAcc[epoch] = int(val_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
                test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
                testAcc[epoch] =  int(test_correct.sum()) / int(data.test_mask.sum()) 
                #valAcc[epoch] = roc_auc_score(data.y[data.val_mask].cpu(),pred[data.val_mask].cpu())
                #testAcc[epoch] = roc_auc_score(data.y[data.test_mask].cpu(),pred[data.test_mask].cpu())
            
            if saveWeightsAtMaxValAcc and valAcc[epoch]>maxValAcc:
                paramsAtMaxValAcc  = deepCopyParamsToNumpy(params)
                maxValAcc = valAcc[epoch]

            if(trainLoss[epoch]<trainLossToConverge):
                continueTraining=False

            if lrDecayFactor<1:
                lrScheduler.step(valAcc[epoch])

            #implement loop for early stopping based on val loss or val acc or both? Or simply train till convergence and later find test acc at min val acc
            # if earlyStop:
            #     if early_stopper.early_stop(valLoss[epoch]):   
            #     continueTraining=False

                #torch.nn.ParameterList() not needed in this case because the parameter values/grads are copied from tensors to numpy

            #section needed if alpha values are recorded 
            # for l in range(numLayers):
            #     a=computeAlphaStatSumry(attnCoef[l],quantiles)
            #     for x in labels:
            #     alphaStatSumry[l]['alpha_ii'][x][epoch] = a[0][x]
            #     alphaStatSumry[l]['alpha_ij'][x][epoch] = a[1][x]
            #     #params[epoch][l]['alphas'] = attnCoef#.detach().cpu().numpy()

            #record quantities again for the gradients in the epoch and change in weights as a result of it
            

            # for l in range(numLayers):
            #     for p in paramTypes:
            #         #currRec[l][p]['wght'] = params[l][p].data.detach().cpu().numpy()
            #         #currRec[l][p]['grad'] = params[l][p].grad.detach().cpu().numpy()
            #         grads=params[l][p].grad.detach()
            #         #if p=='attn':
            #             #currRec[l][p]['wght'] = currRec[l][p]['wght'][0]
            #             #currRec[l][p]['grad'] = currRec[l][p]['grad'][0]
            #         for k,v in computeStatSumry(grads,quantiles).items():
            #             paramStatSumry[l][p]['grad'][k][epoch] = v

            # for l in range(numLayers):
            #     for p in paramTypes:
            #         for k,v in computeStatSumry(abs((currRec[l][p]['wght']-prevRec[l][p]['wght'])/prevRec[l][p]['wght']),quantiles).items():
            #             changeInParamStatSumry[l][p]['wght'][k][epoch]=v
                
            #prevRec = copy.deepcopy(currRec) #if prevRec is (initially) set outside training loop with initialized value 

            if(epoch%printLossEveryXEpoch==0 or epoch==numEpochs-1):
                print(f'--Epoch: {epoch:03d}, Train Loss: {loss:.4f}')
            epoch+=1

        finalParamsCopy  = deepCopyParamsToNumpy(params)

        trainLoss = trainLoss[:epoch].detach().cpu().numpy()
        valLoss = valLoss[:epoch].detach().cpu().numpy()
        trainAcc = trainAcc[:epoch].detach().cpu().numpy()
        valAcc = valAcc[:epoch].detach().cpu().numpy()
        testAcc = testAcc[:epoch].detach().cpu().numpy()
        
        print('Max or Convergence Epoch: ', epoch)
        print('Max Validation Acc At Epoch: ', np.argmax(valAcc)+1)
        print('Test Acc at Max Val Acc:', testAcc[np.argmax(valAcc)]*100)

        if saveParamGradStatSumry:
            for l in range(numLayers):
                for p in paramTypes:
                    for x in labels:
                        paramStatSumry[l][p]['wght'][x] = paramStatSumry[l][p]['wght'][x][:epoch].cpu().numpy()
                        paramStatSumry[l][p]['grad'][x] = paramStatSumry[l][p]['grad'][x][:epoch].cpu().numpy()

        if saveNeuronLevelL2Norms:
            for l in range(numLayers):
                attnWghtsSq[l] = attnWghtsSq[l][0:epoch,:].T.cpu().numpy()
                for p in set(paramTypes)-set(['attn']):
                    featL2Norms[l][p]['row'] = featL2Norms[l][p]['row'][0:epoch,:].T.cpu().numpy()
                    featL2Norms[l][p]['col'] = featL2Norms[l][p]['col'][0:epoch,:].T.cpu().numpy()

        if saveLayerWiseForbNorms:
            for l in range(numLayers):
                for p in paramTypes:
                    forbNorms[l][p]['wght'] = forbNorms[l][p]['wght'][:epoch].cpu().numpy()
                    forbNorms[l][p]['grad'] = forbNorms[l][p]['grad'][:epoch].cpu().numpy()


        #this section is not really necessary unless early stopping is implemented and only recorded values for trainedEpochs<maxEpochs need to be saved
        # for l in range(numLayers):
        #     attnWghtsSq[l] = attnWghtsSq[l][0:epoch,:].T.cpu().numpy()
        #     for p in set(paramTypes)-set(['attn']):
        #         featL2Norms[l][p]['row'] = featL2Norms[l][p]['row'][0:epoch,:].T.cpu().numpy()
        #         featL2Norms[l][p]['col'] = featL2Norms[l][p]['col'][0:epoch,:].T.cpu().numpy()
            # for x in labels:
            #     # alphaStatSumry[l]['alpha_ii'][x] = alphaStatSumry[l]['alpha_ii'][x][:epoch]
            #     # alphaStatSumry[l]['alpha_ij'][x] = alphaStatSumry[l]['alpha_ij'][x][:epoch]
            #     for p in paramTypes:
            #         paramStatSumry[l][p]['wght'][x] = paramStatSumry[l][p]['wght'][x][:epoch].cpu().numpy()
            #         paramStatSumry[l][p]['grad'][x] = paramStatSumry[l][p]['grad'][x][:epoch].cpu().numpy()
            #         changeInParamStatSumry[l][p]['wght'][x] = changeInParamStatSumry[l][p]['wght'][x][:epoch]

        
        expDict = {'expID':expID,  
                'trainedEpochs':epoch,
                'trainLoss':trainLoss,
                'valLoss':valLoss,
                'trainAcc':trainAcc,
                'valAcc':valAcc,
                'testAcc':testAcc,
                'initialParams':initialParamsCopy,
                'finalParams':finalParamsCopy,
                'paramsAtMaxValAcc':paramsAtMaxValAcc
        }

        with open(path+'dictExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
            pickle.dump(expDict,f)

        if saveParamGradStatSumry:
            saveParamStatSumry = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'quantiles':quantiles.cpu().numpy(),
                        'statSumry':paramStatSumry
                    }
            with open(path+'paramStatSumryExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveParamStatSumry,f)

        if saveNeuronLevelL2Norms:
            saveNeuronLevelAttnAndFeatL2Norms = {
                        'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'featL2Norms':featL2Norms,
                        'attnWghtsSq':attnWghtsSq
                    }
            with open(path+'neuronLevelAttnAndFeatL2Norms'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveNeuronLevelAttnAndFeatL2Norms,f)

        if saveLayerWiseForbNorms:
            saveForbNorms = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'forbNorms':forbNorms
                }
            with open(path+'forbNormsExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveForbNorms,f)

        # saveAlphaStatSumry = {'expID':expID,
        #                 'numLayers':numLayers,
        #                 'trainedEpochs':epoch,
        #                 'quantiles':quantiles,
        #                 'statSumry':alphaStatSumry
        #         }
        # saveChangeInparamStatSumry = {
        #                 'expID':expID,
        #                 'numLayers':numLayers,
        #                 'trainedEpochs':epoch,
        #                 'quantiles':quantiles,
        #                 'statSumry':changeInParamStatSumry
        # }


        # with open(path+'alphaStatSumryExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
        #         pickle.dump(saveAlphaStatSumry,f)
        
        # with open(path+'changeInParamStatSumryExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
        #     pickle.dump(saveChangeInparamStatSumry,f)
        #torch.save(model,path+'modelExp'+str(expID)+'_run'+str(run)+'.pt')

