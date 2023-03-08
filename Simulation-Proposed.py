# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from skorch.net import NeuralNet
from model.model import OneStageHidden, MMS
import torch.nn as nn
import pandas as pd
from skorch.net import NeuralNet
from model.net import Net
import torch
import numpy as np
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from sklearn.metrics import explained_variance_score
from argparse import ArgumentParser
import itertools
import sys
import skorch
from skorch.callbacks import EpochScoring, EarlyStopping

#sys.argv = ['main.py']

nstage = 9
ninput = 75
noutput = 6
otherinput = 15
hidden_size = 10
nonactivate = None
noiselevel = 0.5
Ntrain = 500
Nvalidation = 500
hidden2_size = 0
Ntest = 2000
isindi = False
l1weight = 0.001
l2weight = 0.0001
lr = 0.01
icase = 1
seed = 0



def get_args():
    parser = ArgumentParser(description='PyTorch MMS Simulation')
    parser.add_argument('--experiment', type=str, default='case1')    
    parser.add_argument('--hidden_size', type=int, default=10)
    parser.add_argument('--hidden2_size', type=int, default=0)
    parser.add_argument('--l1weight', type=float, default=.001)
    parser.add_argument('--ol1weight', type=float, default=.000)
    parser.add_argument('--l2weight', type=float, default=0.000001)
    parser.add_argument('--ol2weight', type=float, default=0.000)
    parser.add_argument('--nonactivate', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--isindi', type=bool, default=False)
    parser.add_argument('--inidimodel', type=bool, default=False)
    args = parser.parse_args()
    return args
args = get_args()

seed = args.seed
N = Ntrain + Ntest + Nvalidation

def gensimulatedata(N,nstage=10,ninput=80,noutput=6,hidden_size=10,isindi=False,otherinput=20,nonactivate=None,noiselevel=0.5,hidden2_size=0,seed=seed):
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


def splitdata(Xdataall_t,ytrue_t,Ntrain,Ntest,seed=seed):
    Xtrain_t = Xdataall_t[:Ntrain,:]
    ytrain_t = ytrue_t[:Ntrain]
    ytrain_t = torch.Tensor(ytrain_t.data).float() + noiselevel * torch.randn(ytrain_t.shape)
    Xtest_t = Xdataall_t[Ntrain:(Ntrain+Ntest),:]
    ytest_t = ytrue_t[Ntrain:(Ntrain+Ntest)]
    
    return Xtrain_t,ytrain_t,Xtest_t,ytest_t


# +




run_configs = {'hidden_size': [5,10,30],'hidden2_size':[0,5,10],'l1weight':[0.1,0.01,0.001],'inidimodel':[True,False]}
keys, values = zip(*run_configs.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]


experiment =args.experiment
seed = args.seed

# Case 1
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
    
Xtrain_t,ytrain_t,Xtest_t,ytest_t = splitdata(Xdataall_t,ytrue_t,Ntrain+Nvalidation,Ntest)
Xtrain_t,ytrain_t,Xval_t,yval_t = splitdata(Xtrain_t,ytrain_t,Ntrain,Nvalidation)
valid_ds = Dataset(Xval_t.cuda(), yval_t.cuda())

len(experiments)



for i,val in enumerate(experiments):
    print(i)
    args.hidden_size = val['hidden_size']
    args.hidden2_size = val['hidden2_size']
    args.l1weight = val['l1weight']
    args.inidimodel = val['inidimodel']
    globals().update(args.__dict__)

    filename = '_'.join("{}".format(val) for (key,val) in args.__dict__.items())
    filename = 'simulation/Simulation_MMS_'+filename

    filenamepkl = filename +'.pkl'
    print(filename)
    logfile = '_'.join([experiment,str(nonactivate),str(seed)])
    logfile = 'simulation/'+logfile

    R2 = EpochScoring(scoring='explained_variance', lower_is_better=False)
    
    filenamelog = filename +'.log'
    old_stdout = sys.stdout
    log_file = open(filenamelog,"w")
    sys.stdout = log_file

    class save_best_to_file(skorch.callbacks.Callback):
        def on_epoch_end(self, net, **kwargs):
            if net.history[-1]['valid_loss_best']:
                net.save_params(filenamepkl)
    #            net.save_history(filenamehistory)

    earlystopping = EarlyStopping(monitor='valid_loss', patience=100, threshold=0.0001, threshold_mode='rel', lower_is_better=True)

    net = Net(
        module=MMS,
        max_epochs = 3000,
        batch_size=128,
        lr = lr,
        module__nstage = nstage,
        module__hidden_size = hidden_size,
        module__hidden2_size = hidden2_size,
        l1weight=l1weight,
        l2weight=l2weight,        
        ol1weight=ol1weight,
        ol2weight=ol2weight,
        module__nonlinearity = nonactivate,
        module__includeoutput = False,
        module__indi=inidimodel,
        module__train_columns = Xdataall_pd.columns, 
        module__test_continuous_columns = yall_p.columns,
        module__test_discrete_columns = None, 
        module__outputstage =range(nstage),
        module__process_keyword = ['f'],
        device='cuda',
        callbacks=[R2,save_best_to_file,earlystopping],
        criterion = nn.MSELoss,
        optimizer = torch.optim.Adam,
    #    optimizer = torch.optim.SGD,
        train_split=predefined_split(valid_ds),
        optimizer_momentum = 0.95,
        warm_start=True,
        iterator_train__shuffle=True
    )


    net.fit(Xtrain_t.cuda(),ytrain_t.cuda())
    y_test_pred = net.predict(Xtest_t)
    explained_variance_score(ytest_t.detach().numpy(),y_test_pred) 
    net.initialize()  # This is important!
    net.lr = 0.001
    net.load_params(filenamepkl)
    net.fit(Xtrain_t.cuda(),ytrain_t.cuda())

    net.initialize()  # This is important!
    net.lr = 0.0001
    net.load_params(filenamepkl)
    net.fit(Xtrain_t.cuda(),ytrain_t.cuda())

    net.initialize()  # This is important!
    net.optimizer = torch.optim.SGD
    net.lr = 0.001
    net.load_params(filenamepkl)
    net.fit(Xtrain_t.cuda(),ytrain_t.cuda())
    net.initialize()  # This is important!
    net.optimizer = torch.optim.SGD
    net.lr = 0.0001
    net.load_params(filenamepkl)
    net.fit(Xtrain_t.cuda(),ytrain_t.cuda())


    net.initialize()  # This is important!
    net.load_params(filenamepkl)
    a=np.sum((ytest_t.data.numpy()-y_test_pred)**2,0)/np.sum((ytest_t.data.numpy()-np.mean(ytrain_t.data.numpy(),0))**2,0)
    epscore = np.array(a)
    
    
    with open(logfile, "a") as myfile:
        strlog = filenamepkl+"{} ({})".format(np.mean(epscore),np.std(epscore))
        logfile.write(logfile)
    print(strlog)
    log_file.close()
# -




