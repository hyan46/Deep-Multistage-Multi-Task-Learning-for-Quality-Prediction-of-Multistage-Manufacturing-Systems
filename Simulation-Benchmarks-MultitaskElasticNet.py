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
from gensimulation import gen_simulation
import sys
from sklearn import linear_model

sys.argv = ['main.py']

# +
nstage = 9
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--isindi', type=bool, default=False)
    args = parser.parse_args()
    return args

Xdataall_pd, Xtrain_t,ytrain_t,Xval_t,yval_t, Xtest_t,ytest_t,valid_ds,levelf,levelv = gen_simulation('case3', Ntrain,Nvalidation,Ntest)


args = get_args()
globals().update(args.__dict__)
allexperiments = ['case1','case2','case3']

clfall = [[]  for i in allexperiments]


epscore = [[] for i in allexperiments]

Xtrain_pd = pd.DataFrame(data = Xtrain_t.data.numpy(), columns =  levelf)
Xtest_pd = pd.DataFrame(data = Xtest_t.data.numpy(), columns =  levelf)


# +

for iexperiment,experiment in enumerate(allexperiments):
    N = Ntrain + Ntest + Nvalidation
    Xdataall_pd, Xtrain_t,ytrain_t,Xval_t,yval_t, Xtest_t,ytest_t,valid_ds,levelf,levelv = gen_simulation(experiment, Ntrain,Nvalidation,Ntest)
    ytrain_pd = pd.DataFrame(data = ytrain_t.data.numpy(), columns =  levelv)
    ytest_pd= pd.DataFrame(data = ytest_t.data.numpy(), columns =  levelv)

    for istage in range(nstage):
        clf = linear_model.MultiTaskElasticNetCV(l1_ratio=[0.01, .1, .5, .7, .9, .95, .99, 1],alphas = [0.01,.1, 1., 2., 4.,10.,100])
        Xtrain_pd_reg = pd.concat(tuple([Xtrain_pd['f'][str(istage)] for istage in range(istage+1)]), axis=1)
        Xtest_pd_reg = pd.concat(tuple([Xtest_pd['f'][str(istage)] for istage in range(istage+1)]), axis=1)
        ytrain = ytrain_pd[str(istage)].values
        ytest = ytest_pd[str(istage)].values


        clf.fit(Xtrain_pd_reg.values,ytrain)
        y_pred = clf.predict(Xtest_pd_reg.values)
        score = np.sum((ytest-y_pred)**2,0)/np.sum((ytest-np.mean(ytrain))**2,0)
        print(istage,score)
        epscore[iexperiment].append(score)
        
    clfall[iexperiment] = clf
# -

import pickle
with open('MMS_simulation_MultiElasticNet.pickle','wb') as f:
    pickle.dump([clfall,epscore],f)

for i in range(3):
     print("{:.3f} ({:.3f})".format(np.mean(epscore[i]),np.std(epscore[i])))
    

discovernum = [ [] for i in range(3)]
allindex = np.arange(810)
for iexperiment,experiment in enumerate(allexperiments):
    if experiment == 'case1':
        informative = np.concatenate(tuple([np.arange(90*j+0,90*j+75) for j in range(9)]))
        noninformative = np.setdiff1d(allindex,informative)
    elif experiment == 'case2':
        informative = np.concatenate(tuple([np.arange(90*j+60,90*j+85) for j in range(9)]))
        noninformative  = np.concatenate([np.arange(25,30),np.arange(55,60),np.arange(85,90)])    
    else:
        s1 = np.arange(180,(180+75))
        s2 = np.arange(5*90,(5*90+75))
        s3 = np.arange(8*90,(8*90+75))
        informative = np.concatenate((s1,s2,s3))
        noninformative = np.setdiff1d(allindex,informative)
    score = np.abs(clf.coef_[5])
    discovernum[iexperiment] = (np.sum(score[informative]>np.percentile(score[noninformative],95)))/len(informative)


discovernum

import matplotlib.pyplot as plt
plt.plot(clf.coef_.T)
plt.show()

# +
from sklearn.metrics import auc

clf.coef_[5]
# -



discovernum


