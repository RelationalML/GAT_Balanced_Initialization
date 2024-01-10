import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from math import ceil

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        initNorm = 2
        initDefault = False
        baseLLortho = True
        self.numLayers = 2
        zeroA=False
        self.dropout = dropout
        initW = [None] * self.numLayers
        if not initDefault:
            zeroA=True
            initW[0] = torch.empty(size=(nhid*nheads,nfeat))
            for l in range(1,self.numLayers-1):
                initW[l]=torch.empty(size=(nhid*nheads,nhid*nheads))
            initW[self.numLayers-1] = torch.empty(size=(nclass,nhid*nheads))
            if not baseLLortho:
                for l in range(self.numLayers):
                    nn.init.xavier_normal_(initW[l]) 
            else:
                dim0 = (ceil(initW[0].shape[0]/2),initW[0].shape[1])
                firstLayerDelta = torch.nn.init.orthogonal_(torch.empty(dim0[0],dim0[1]))
                initW[0] = torch.cat((firstLayerDelta,-firstLayerDelta),dim=0)
                for l in range(1,self.numLayers-1):
                    delta = nn.init.orthogonal_(torch.empty(ceil(initW[l].shape[0]/2),ceil(initW[l].shape[1]/2)))
                    delta = torch.cat((delta, -delta), dim=0)
                    delta = torch.cat((delta, -delta), dim=1)
                    initW[l] = delta
                dimL = (initW[self.numLayers-1].shape[0],ceil(initW[self.numLayers-1].shape[1]/2))
                finalLayerDelta = torch.nn.init.orthogonal_(torch.empty(dimL[0],dimL[1]))
                initW[self.numLayers-1] = torch.cat((finalLayerDelta,-finalLayerDelta),dim=1)
            
            incSqNorm = torch.sqrt(torch.pow(initW[0],2).sum(axis=1))
            reqRowWiseSqL2Norm = torch.full((initW[0].size()[0],),float(initNorm))
            initW[0]=torch.multiply(torch.divide(initW[0],incSqNorm.reshape((len(incSqNorm),1))),\
                        torch.sqrt(reqRowWiseSqL2Norm.reshape(len(reqRowWiseSqL2Norm),1)))
            
            for l in range(1,self.numLayers):
                # print(l)
                # print(initW[l-1].shape, ' : ',initW[l].shape)
                incSqNorm = torch.pow(initW[l-1],2).sum(axis=1)
                outSqNorm = torch.sqrt(torch.pow(initW[l],2).sum(axis=0))
                initW[l] = torch.multiply(torch.divide(initW[l],outSqNorm.reshape((1,len(outSqNorm)))),\
                                            torch.sqrt((incSqNorm).reshape((1,len(incSqNorm)))))
                
            for l in range(self.numLayers):
                initW[l] = initW[l].T
        
        self.attentions = [None]*self.numLayers
        for l in range(self.numLayers-1):
            if l==0:
                inFeat=nfeat
            else:
                inFeat=nhid
            subW = [None] * nheads
            if initW[l]!=None:
                for i in range(nheads):
                    subW[i]=initW[l][:,(i*nhid):((i+1)*nhid)]
           # for i in range(nheads):
            #    print(l,' : ',i,' : ',initW[l][:,(i*nhid):((i+1)*nhid)].shape)
            self.attentions[l] = [GraphAttentionLayer(subW[i],zeroA,inFeat, nhid, dropout=dropout, alpha=alpha, concat=True) for i in range(nheads)]
            for i, attention in enumerate(self.attentions[l]):
                self.add_module('attention_{}_{}'.format(l,i), attention)

        self.out_att = GraphAttentionLayer(initW[self.numLayers-1],zeroA,nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        for l in range(self.numLayers-1):
            x = torch.cat([att(x, adj) for att in self.attentions[l]], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

