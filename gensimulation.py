import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser

from skorch.dataset import Dataset

from model.model import OneStageHidden, MMS

def gensimulatedata(N,nstage=10,ninput=80,noutput=6,hidden_size=10,isindi=False,otherinput=20,nonactivate=None,noiselevel=0.5,hidden2_size=0,seed=0):
    np.random.seed(seed)
    input_size = [ninput for i in range(nstage)]
    output_size =[noutput for i in range(nstage)]
    input_size_all = [ninput + otherinput for i in range(nstage)]

    level1f = ['f' for i in range(nstage*ninput)]
    level2f = [str(i) for i in range(nstage)  for j in range(ninput)]
    level3f = [i for i in range(nstage*ninput)]
    Xdata_pd = pd.DataFrame(data = np.random.randn(N,nstage*ninput), columns =  [level1f,level2f,level3f])
    index = Xdata_pd.columns

    level1v = [str(i) for i in range(nstage)  for j in range(noutput)]
    level2v = [i for i in range(nstage*noutput)]

    ydata_columns = pd.DataFrame(data = np.random.randn(N,nstage*noutput), columns =  [level1v,level2v]).columns
    Xdata_t = torch.from_numpy(Xdata_pd.values).float()
    MMStrue = MMS(nstage, hidden_size, train_columns = index, test_continuous_columns=ydata_columns,test_discrete_columns=None, hidden2_size = 0, process_keyword=['f'],quality_keywords=['v'], nonlinearity=None, includeoutput=False, initialize=[None], usecuda=False, indi=isindi, outputstage=range(nstage))
    ytrue_t = MMStrue.forward(Xdata_t)
    
    if otherinput != 0:
        level1f = ['f' for i in range(nstage*(ninput+otherinput))]
        level2f = [str(i) for i in range(nstage)  for j in range(ninput+otherinput)]
        level3f = [i for i in range(nstage*(ninput+otherinput))]

        Xall = np.concatenate(tuple([np.concatenate(((Xdata_pd['f'][str(istage)].values),np.random.randn(N,otherinput)), axis=1) for istage in range(nstage)]),axis=1)
        Xdataall_pd = pd.DataFrame(data = Xall, columns =  [level1f,level2f,level3f])
        Xdataall_t = torch.from_numpy(Xdataall_pd.values).float()
    else:
        Xdataall_t = Xdata_t
        Xdataall_pd = Xdata_pd
        
    yall_pd = pd.DataFrame(data=ytrue_t.data.numpy(), columns = ydata_columns)

    return Xdataall_t,ytrue_t,Xdataall_pd,yall_pd


def splitdata(Xdataall_t,ytrue_t,Ntrain,Ntest,noiselevel,seed):
    Xtrain_t = Xdataall_t[:Ntrain,:]
    ytrain_t = ytrue_t[:Ntrain]
    ytrain_t = torch.Tensor(ytrain_t.data).float() + noiselevel * torch.randn(ytrain_t.shape)
    Xtest_t = Xdataall_t[Ntrain:(Ntrain+Ntest),:]
    ytest_t = ytrue_t[Ntrain:(Ntrain+Ntest)]
    
    return Xtrain_t,ytrain_t,Xtest_t,ytest_t


def gen_simulation(experiment,Ntrain,Nvalidation,Ntest,nstage=9,ninput=75,noutput=6,hidden_size=10,isindi=False,otherinput=15,nonactivate=None,noiselevel=0.5,hidden2_size=0,seed=0):
    # Case 1
    N = Ntrain + Ntest + Nvalidation
    level1f = ['f' for i in range(nstage*(ninput+otherinput))]
    level2f = [str(i) for i in range(nstage)  for j in range((ninput+otherinput))]
    level3f = [i for i in range(nstage*(ninput+otherinput))]

    level1v = [str(i) for i in range(nstage)  for j in range(noutput)]
    level2v = [i for i in range(nstage*noutput)]

    if experiment == 'case1':
        # Case 1: 1 unified Line
        Xdataall_t,ytrue_t,Xdataall_pd,yall_p = gensimulatedata(N,nstage,ninput,noutput,hidden_size,isindi,otherinput,nonactivate,noiselevel,hidden2_size,seed=seed)
    elif experiment == 'case2':
       # Case 2: 3 sensor groups
        nLine = 3
        Xdataall_pd_Line = [[] for i in range(nLine)]
        yall_pd_Line = [[] for i in range(nLine)]
        ninputsingle = int(ninput/nLine)
        noutputsingle = int(noutput/nLine)
        otherinputsingle = int(otherinput/nLine)
        for j in range(nLine):
            a,b,Xdataall_pd_Line[j],yall_pd_Line[j] = gensimulatedata(N,nstage,ninputsingle,noutputsingle,hidden_size,isindi,otherinputsingle,nonactivate,noiselevel,hidden2_size,seed=seed+j)
        Xdataall_list = [[] for istage in range(nstage)]
        yall_list = [[] for istage in range(nstage)]
        for istage in range(nstage):
            ydata_stage_tuple = tuple([yall_pd_Line[j][str(istage)] for j in range(nLine)])
            Xdata_stage_tuple = tuple([Xdataall_pd_Line[j]['f'][str(istage)] for j in range(nLine)])
            Xdataall_list[istage] = pd.concat(Xdata_stage_tuple,1)
            yall_list[istage] = pd.concat(ydata_stage_tuple,1)


        Xdataall_pd_single = pd.concat(tuple(Xdataall_list),1)
        yall_pd_single = pd.concat(tuple(yall_list),1)

        Xdataall_pd = pd.DataFrame(data = Xdataall_pd_single.values, columns =  [level1f,level2f,level3f])
        yall_pd = pd.DataFrame(data = yall_pd_single.values, columns =  [level1v,level2v])

        Xdataall_t = torch.from_numpy(Xdataall_pd.values).float()
        ytrue_t = torch.from_numpy(yall_pd.values).float()
    else:    
        nLine = 3
        Xdataall_pd_Line = [[] for i in range(nLine)]
        yall_pd_Line = [[] for i in range(nLine)]
        nstagesingle = int(nstage/nLine)
        for j in range(nLine):
            a,b,Xdataall_pd_Line[j],yall_pd_Line[j] = gensimulatedata(N,nstagesingle,ninput,noutput,hidden_size,isindi,otherinput,nonactivate,noiselevel,hidden2_size,seed=seed+j)
        Xdataall_tuple = tuple([Xdataall_pd_Line[j]['f'][str(istage)] for istage in range(nstagesingle) for j in range(nLine)])
        Xdataall_single = pd.concat(Xdataall_tuple,1)
        yall_pd_tuple = tuple([yall_pd_Line[j][str(istage)] for istage in range(nstagesingle) for j in range(nLine)])
        yall_pd_single = pd.concat(yall_pd_tuple,1)
        Xdataall_pd = pd.DataFrame(data = Xdataall_single.values, columns =  [level1f,level2f,level3f])
        yall_pd = pd.DataFrame(data = yall_pd_single.values, columns =  [level1v,level2v])
        ytrue_t = torch.from_numpy(yall_pd.values).float()
    levelf =  [level1f,level2f,level3f]
    levelv =  [level1v,level2v]

    Xdataall_t = torch.from_numpy(Xdataall_pd.values).float()
    Xtrain_t,ytrain_t,Xtest_t,ytest_t = splitdata(Xdataall_t,ytrue_t,Ntrain+Nvalidation,Ntest,noiselevel,seed)
    Xtrain_t,ytrain_t,Xval_t,yval_t = splitdata(Xtrain_t,ytrain_t,Ntrain,Nvalidation,noiselevel,seed)
    valid_ds = Dataset(Xval_t.cuda(), yval_t.cuda())
    return Xdataall_pd,Xtrain_t,ytrain_t,Xval_t,yval_t, Xtest_t,ytest_t,valid_ds,levelf,levelv


