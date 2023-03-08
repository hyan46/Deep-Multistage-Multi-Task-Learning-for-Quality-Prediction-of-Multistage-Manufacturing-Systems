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
from sklearn import datasets, linear_model, ensemble
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

sys.argv = ['main.py']
# -

# ## Benchmark Methods

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


args = get_args()
globals().update(args.__dict__)

allmodel = [linear_model.LinearRegression(),
            linear_model.ElasticNetCV(fit_intercept=True, alphas=[0.01,.1, 1., 2., 4.], l1_ratio= [0.01,0.1,0.5,0.9,0.99], cv = 3), 
            ensemble.RandomForestRegressor(n_estimators = 100, min_samples_split= 5, min_samples_leaf =2, max_features='auto',max_depth=10), 
            MLPRegressor(hidden_layer_sizes=(10, ))] 

allexperiments = ['case1','case2','case3']
epscore = [[[] for i in allmodel] for i in allexperiments]



# -

# ## Load Data

# Xdataall_pd, Xtrain_t,ytrain_t,Xval_t,yval_t, Xtest_t,ytest_t,valid_ds,levelf,levelv = gen_simulation(experiment, Ntrain,Nvalidation,Ntest)
print('loading data')
import pickle
iexperiment = int(experiment[4])
with open("simu_{}.pkl".format(iexperiment), "rb")  as f:
    [Xdataall_pd, Xtrain_t,ytrain_t,Xval_t,yval_t, Xtest_t,ytest_t,valid_ds,levelf,levelv] = pickle.load(f)

# ### Loading Pre-trained Model

import pickle
print('loading Model')
clfprop = [[] for i in range(3)]
epprop =  [[] for i in range(3)]
for iexperiment,experiment in enumerate(allexperiments):
    if experiment == 'case1':
        filename = 'savedmodel/Simulation_MMS_case1_10_0_0.001_0.01_0.0001_0.0001_None_0_False_True.pkl'
    elif experiment == 'case2':
        filename = 'savedmodel/Simulation_MMS_case2_30_0_0.01_0.1_0.001_0.0001_None_0_False_True.pkl'
    else:
        filename = 'savedmodel/Simulation_MMS_case3_30_0_0.03_0.001_0.0001_0.1_None_0_False_True.pkl'


    hidden_size = int(filename.split('_')[3])
    hidden2_size = int(filename.split('_')[4])
    indi = filename.split('_')[12][0] == 'T'
    ytrain_pd = pd.DataFrame(data = ytrain_t.data.numpy(), columns =  levelv)
    ytest_pd= pd.DataFrame(data = ytest_t.data.numpy(), columns =  levelv)

    net = Net(
        module=MMS,
        max_epochs = 3000,
        batch_size=128,
        lr = lr,
        module__nstage = nstage,
        module__hidden_size = hidden_size,
        module__hidden2_size = hidden2_size,
        l1weight=0,
        l2weight=0,        
        ol1weight=0,
        ol2weight=0,
        module__nonlinearity = nonactivate,
        module__includeoutput = False,
        module__indi=indi,
        module__train_columns = Xdataall_pd.columns, 
        module__test_continuous_columns = ytrain_pd.columns,
        module__test_discrete_columns = None, 
        module__outputstage =range(nstage),
        module__process_keyword = ['f'],
        device='cuda',
        criterion = nn.MSELoss,
        optimizer = torch.optim.Adam,
    #    optimizer = torch.optim.SGD,
        train_split=predefined_split(valid_ds),
        optimizer_momentum = 0.95,
        warm_start=True,
        iterator_train__shuffle=True
    )
    net.initialize()
    net.load_params(filename)
    clfprop[iexperiment] = net

# +
# for iexperiment,experiment in enumerate(allexperiments):
#     iexperiment = int(experiment[4])
#     Xdataall_pd, Xtrain_t,ytrain_t,Xval_t,yval_t, Xtest_t,ytest_t,valid_ds,levelf,levelv = gen_simulation(experiment, Ntrain,Nvalidation,Ntest)
#     with open("simu_{}.pkl".format(iexperiment), "wb")  as f:
#         pickle.dump([Xdataall_pd, Xtrain_t,ytrain_t,Xval_t,yval_t, Xtest_t,ytest_t,valid_ds,levelf,levelv],f,protocol=pickle.HIGHEST_PROTOCOL)
# -
# ### Accuracy


print('Accuracy')
Xtrain_tall = [[] for i in range(3)]
for i,experiment in enumerate(allexperiments):
    iexperiment = int(experiment[4])
    with open("simu_{}.pkl".format(iexperiment), "rb")  as f:
        [Xdataall_pd, Xtrain_t,ytrain_t,Xval_t,yval_t, Xtest_t,ytest_t,valid_ds,levelf,levelv] = pickle.load(f)
    Xtrain_tall[i] = Xtrain_t
    y_test_pred = clfprop[i].predict(Xtest_t)
    a=np.sum((ytest_t.data.numpy()-y_test_pred)**2,0)/np.sum((ytest_t.data.numpy()-np.mean(ytrain_t.data.numpy(),0))**2,0)
    epscore = np.array(a)
    epprop[i] = epscore
    print(np.mean(epscore))
    print(np.std(epscore))

# ### Plot the Important sensors for the final sensor
# Plot important sensors of the last sensor 


# +
from sklearn.metrics import auc
from skorch.utils import to_numpy

experiment = 'case2'
iexperiment = int(experiment[4])-1
print(iexperiment)


# -

# ## Get the list of important sensors

def get_sens_idx(experiment):
    allindex = np.arange(810)
    if experiment == 'case1':
        informative = np.concatenate(tuple([np.arange(90*j+0,90*j+75) for j in range(9)]))
        noninformative = np.setdiff1d(allindex,informative)
    elif experiment == 'case2':
        informative = np.concatenate(tuple([np.arange(90*j+60,90*j+85) for j in range(9)]))
        noninformative  = np.setdiff1d(allindex,informative)
    else:
        s1 = np.arange(180,(180+75))
        s2 = np.arange(5*90,(5*90+75))
        s3 = np.arange(8*90,(8*90+75))
        informative = np.concatenate((s1,s2,s3))
        noninformative = np.setdiff1d(allindex,informative)
    return informative,noninformative



# ## Training of all benchmarks

# +
istrain = False
issave = False  

clfall = [[] for i in range(len(allmodel)+1)] 
epscore = [[] for i in range(len(allmodel)+1)] 
informative,noninformative = get_sens_idx(experiment)

with open("simu_{}.pkl".format(iexperiment+1), "rb")  as f:
    [Xdataall_pd, Xtrain_t,ytrain_t,Xval_t,yval_t, Xtest_t,ytest_t,valid_ds,levelf,levelv] = pickle.load(f)

N = Ntrain + Ntest + Nvalidation
Xtrain_pd = pd.DataFrame(data = Xtrain_t.data.numpy(), columns =  levelf)
Xtest_pd =  pd.DataFrame(data = Xtest_t.data.numpy(), columns =  levelf)
ytrain_pd = pd.DataFrame(data = ytrain_t.data.numpy(), columns =  levelv)
ytest_pd= pd.DataFrame(data = ytest_t.data.numpy(), columns =  levelv)

N = Ntrain + Ntest + Nvalidation
Xtrain_pd = pd.DataFrame(data = Xtrain_t.data.numpy(), columns =  levelf)
Xtest_pd =  pd.DataFrame(data = Xtest_t.data.numpy(), columns =  levelf)
ytrain_pd = pd.DataFrame(data = ytrain_t.data.numpy(), columns =  levelv)
ytest_pd= pd.DataFrame(data = ytest_t.data.numpy(), columns =  levelv)

dfregall_pd = pd.concat(tuple([Xdataall_pd['f'][str(istage)] for istage in range(9)]), axis=1)

ytrain = ytrain_t.data.numpy()[:,-1]
ytest = ytest_t.data.numpy()[:,-1]
Xtrain_pd_reg = pd.concat(tuple([Xtrain_pd['f'][str(istage)] for istage in range(9)]), axis=1)
Xtest_pd_reg = pd.concat(tuple([Xtest_pd['f'][str(istage)] for istage in range(9)]), axis=1)



if istrain:
    for imodel,model in enumerate(allmodel):
        model.fit(Xtrain_pd_reg.values,ytrain)
        y_pred = model.predict(Xtest_pd_reg.values)
        score = np.sum((ytest-y_pred)**2)/np.sum((ytest-np.mean(ytrain))**2)
        epscore[imodel].append(score)
        clfall[imodel] = model
        print("{} ({})".format(np.mean(epscore[imodel]),np.std(epscore[imodel])))
    imodel += 1
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
        epscore[imodel].append(score)


    clfall[imodel] = clf


if istrain:
    discovernum = [[] for i in range(6)] 
    scoreall  = [[] for i in range(6)] 
    informativearr = np.zeros(810)

    informativearr[informative] = 1
    for imodel in range(6):
        if imodel <5:
            clf = clfall[imodel]
        else:
            clf = clfprop
        if imodel in [0,1]:
            score = np.abs(clf.coef_)
        if imodel == 4:
            score = np.abs(clf.coef_[5])
        if imodel == 2:
            score = clf.feature_importances_
        if imodel == 3:
            score = np.sum(clf.coefs_[0]**2,1)
        if imodel == 5:
            cX = Xtrain_tall[iexperiment].cuda()
            cX.requires_grad=True
            l = clfprop[iexperiment].module_.forward(cX)
            ressensor = l[:,53].pow(2).sum()
            ressensor.backward()
            cgrad = to_numpy(cX.grad)
            score = np.sum(cgrad**2,0)

        scoreall[imodel] = score

if issave:
    to_be_saved = {
      "model": clfall,
      "feature_importance": scoreall,
      "sensor_number": discovernum}
    with open("result/Simulation_case{}_benchmark_model.pkl".format(iexperiment+1),'wb') as f:
        pickle.dump(to_be_saved,f)
# -


# ## Plot the Important Sensors

# +
isload = True
experiment = 'case3'
iexperiment = int(experiment[4])-1
if isload:
    with open("result/Simulation_case{}_benchmark_model.pkl".format(iexperiment+1),'rb') as f:
        result_to_load = pickle.load(f)


scoreall = result_to_load['feature_importance']
informative,noninformative = get_sens_idx(experiment)
informativearr = np.zeros(810)
informativearr[informative] = 1
import matplotlib 
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
import matplotlib.pyplot as plt
# %matplotlib inline
score_np= np.array(scoreall)
score_np2 = score_np[:,:]
normalized_score = np.zeros((7,810))
normalized_score[:6,:] = score_np2 / np.expand_dims(np.sqrt(np.sum(score_np[:,:]**2,1))*0.1,1)
normalized_score[6,:] = informativearr
normalized_score[normalized_score>1] = 1
plt.figure()
plt.imshow((np.abs(normalized_score[[6,5,4,0,1,2,3],:])),interpolation='nearest', aspect='auto')
plt.xticks(np.arange(0,810,90)+45, ['S{}'.format(i) for i in range(1,10)])
plt.yticks(np.arange(7),['True','DIRMT','MEN','LR','EN','RF','MLP'])
plt.savefig('Figs/SensorIndependentNew_{}.eps'.format(iexperiment+1))
# -
# ### Training for all coefficients

is_training = False
if is_training:
    print('Training')

    import pickle
    oldstage = -99
    clfall = [[[] for i in range(len(allmodel)+1)] for i in allexperiments]
    epscore = [[[] for i in range(len(allmodel)+1)] for i in allexperiments]

    for iexperiment,experiment in enumerate(allexperiments):
        iexp = int(experiment[4])
        with open("simu_{}.pkl".format(iexp), "rb")  as f:
            [Xdataall_pd, Xtrain_t,ytrain_t,Xval_t,yval_t, Xtest_t,ytest_t,valid_ds,levelf,levelv] = pickle.load(f)

        N = Ntrain + Ntest + Nvalidation
        Xtrain_pd = pd.DataFrame(data = Xtrain_t.data.numpy(), columns =  levelf)
        Xtest_pd =  pd.DataFrame(data = Xtest_t.data.numpy(), columns =  levelf)
        ytrain_pd = pd.DataFrame(data = ytrain_t.data.numpy(), columns =  levelv)
        ytest_pd= pd.DataFrame(data = ytest_t.data.numpy(), columns =  levelv)
        i = 0

        for column in ytrain_pd:
            nowstage = int(column[0])
            istage = nowstage
            if nowstage!=oldstage:
                dfregall_pd = pd.concat(tuple([Xdataall_pd['f'][str(istage)] for istage in range(int(column[0])+1)]), axis=1)

            ytrain = ytrain_t.data.numpy()[:,i]
            ytest = ytest_t.data.numpy()[:,i]

            Xtrain_pd_reg = pd.concat(tuple([Xtrain_pd['f'][str(istage)] for istage in range(istage+1)]), axis=1)
            Xtest_pd_reg = pd.concat(tuple([Xtest_pd['f'][str(istage)] for istage in range(istage+1)]), axis=1)

            for imodel,model in enumerate(allmodel):
                model.fit(Xtrain_pd_reg.values,ytrain)
                y_pred = model.predict(Xtest_pd_reg.values)
                score = np.sum((ytest-y_pred)**2)/np.sum((ytest-np.mean(ytrain))**2)
                print(column,score)
                epscore[iexperiment][imodel].append(score)
                clfall[iexperiment][imodel] = model
                print("{} ({})".format(np.mean(epscore[iexperiment][imodel]),np.std(epscore[iexperiment][imodel])))

            oldstage = int(column[1])
            i += 1

        imodel += 1
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
            epscore[iexp-1][imodel].append(score)

        clfall[iexperiment][imodel] = clf





# +
# with open('MMS_simulation_benchmarks_bak_Case2.pickle','wb') as f:
#      pickle.dump(clfall,f)

# with open('MMS_simulation_benchmarks_bak.pickle','rb') as f:
#      [clfall,epscore] = pickle.load(f)

from sklearn.metrics import average_precision_score

with open('result/MMS_simulation_benchmarks_bak.pickle','rb') as f:
     [clfall,epscore] = pickle.load(f)

from sklearn.metrics import auc
from skorch.utils import to_numpy
from sklearn.metrics import precision_recall_curve

allprecision = [[[] for i in range(6)] for i in allexperiments]
allrecall = [[[] for i in range(6)] for i in allexperiments]
all_average_repcision= [[[] for i in range(6)] for i in allexperiments]
allindex = np.arange(810)
scoreall  = [[[] for i in range(6)] for i in allexperiments]

informativearr = [np.zeros(810) for i in range(3)]
for iexperiment,experiment in enumerate(allexperiments):
    
    if experiment == 'case1':
        informative = np.concatenate(tuple([np.arange(90*j+0,90*j+75) for j in range(9)]))
        noninformative = np.setdiff1d(allindex,informative)
    elif experiment == 'case2':
        informative = np.concatenate(tuple([np.arange(90*j+60,90*j+85) for j in range(9)]))
        noninformative  = np.setdiff1d(allindex,informative)
    else:
        s1 = np.arange(180,(180+75))
        s2 = np.arange(5*90,(5*90+75))
        s3 = np.arange(8*90,(8*90+75))
        informative = np.concatenate((s1,s2,s3))
        noninformative = np.setdiff1d(allindex,informative)
    informativearr[iexperiment][informative]=1
    for imodel in range(6):
        if imodel <5:
            clf = clfall[iexperiment][imodel]
        else:
            clf = clfprop[iexperiment]
        if imodel in [0,1]: #LR, EN
            score = np.abs(clf.coef_)
        if imodel == 4: # MEN
            score = np.abs(clf.coef_[5])
        if imodel == 2: # RF
            score = clf.feature_importances_
        if imodel == 3: # MLP
            score = np.sum(clfall[0][3].coefs_[0]**2,1)
        if imodel == 5: #DIRMST
            cX = Xtrain_tall[iexperiment].cuda()
            cX.requires_grad=True
            l = clfprop[iexperiment].module_.forward(cX)
            ressensor = l[:,53].pow(2).sum()
            ressensor.backward()
            cgrad = to_numpy(cX.grad)
            score = np.sum(cgrad**2,0)

        scoreall[iexperiment][imodel] = score
        predict_score = score
        predict_binary = score>np.percentile(score[noninformative],95)
        truelabel= np.zeros(810)
        truelabel[informative] = True
        average_precision = average_precision_score(truelabel, predict_score)
        precision_s, recall_s, _ = precision_recall_curve(truelabel,predict_binary)
        
        allprecision[iexperiment][imodel] = precision_s
        allrecall[iexperiment][imodel] = recall_s
        all_average_repcision[iexperiment][imodel] = average_precision
        
        #(np.sum(score[informative]>np.percentile(score[noninformative],95)))/len(informative)
        
        
# -


for iexperiment,experiment in enumerate(allexperiments):
    print(experiment)
    for imodel in [5,4,0,1,2,3]:
        print(allrecall[iexperiment][imodel])

# +
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
alllegend = ['k.--','bD:']

allmethodlabel = ['MEN','EN','DIRMT']
for iexperiment,experiment in enumerate(allexperiments):
    ii=0
    plt.figure()
    plt.plot(np.mean(epprop[iexperiment].reshape(-1,6),axis=1),'rx-',label="DIRMT")
    for imodel in [4,1]: #[4,0,1,2,3]:
        plt.plot(np.mean(np.array(epscore[iexperiment][imodel]).reshape(-1,6),axis=1),alllegend[ii],label=allmethodlabel[ii])
        ii +=1 

    plt.xticks(np.arange(nstage),np.arange(1, nstage+1, 1))
    plt.xlabel('Stage')
    plt.ylabel('RMSE')
    plt.legend()
#    plt.axis([0.90, 9.1, 0, 0.5])
    plt.show()
    plt.savefig('Figs/{}_RMSE.eps'.format(experiment))
# -




# +
# import pickle
# with open('MMS_simulation_benchmarks_bak.pickle','wb') as f:
#     pickle.dump([clfall,epscore],f)


# -





from sklearn import metrics
aucall = np.zeros((3,6))
for iexperiment,experiment in enumerate(allexperiments):
    for imodel in range(6):
        fpr,tpr,thresholds = metrics.roc_curve(informativearr[iexperiment],scoreall[iexperiment][imodel],pos_label=1)
        aucall[iexperiment,imodel] = metrics.auc(fpr,tpr)

for iexperiment,experiment in enumerate(allexperiments):
    print(experiment)
    for imodel in [5,4,0,1,2,3]:
        print('{:.3f}'.format(aucall[iexperiment][imodel]))












# +
# N = Ntrain + Ntest + Nvalidation
# Xdataall_pd, Xtrain_t,ytrain_t,Xval_t,yval_t, Xtest_t,ytest_t,valid_ds,levelf,levelv = gen_simulation(experiment, Ntrain,Nvalidation,Ntest)
# ytrain_pd = pd.DataFrame(data = ytrain_t.data.numpy(), columns =  levelv)
# ytest_pd= pd.DataFrame(data = ytest_t.data.numpy(), columns =  levelv)
# i = 0
# for column in ytrain_pd:
#     nowstage = int(column[0])
#     if nowstage!=oldstage:
#         dfregall_pd = pd.concat(tuple([Xdataall_pd['f'][str(istage)] for istage in range(int(column[0])+1)]), axis=1)

#     ytrain = ytrain_t.data.numpy()[:,i]
#     ytest = ytest_t.data.numpy()[:,i]

#     dfregtrain_pd = dfregall_pd.iloc[:Ntrain,:]
#     dfregtest_pd = dfregall_pd.iloc[Ntrain:(Ntrain+Ntest),:]

#     for imodel,model in enumerate(allmodel):
#         model.fit(dfregtrain_pd.values,ytrain)
#         y_pred = model.predict(dfregtest_pd.values)
#         score = np.sum((ytest-y_pred)**2)/np.sum((ytest-np.mean(ytrain))**2)
#         print(column,score)
#         epscore[iexperiment][imodel].append(score)

#     oldstage = int(column[1])
#     i += 1

# -



# +
# rf_random.fit(Xtrain_t.numpy(), ytrain_pd[str(nstage-1)].values)


# oldcolumn = '0'
# i = 0

# EL_tuned_parameters = [
#   {'alpha': [0.01, 0.1, 1], 'l1_ratio': [0.1,0.3,0.5,0.7,0.9]}, ]
# clf = GridSearchCV(linear_model.ElasticNet(), EL_tuned_parameters, cv=5,
#                    scoring='neg_mean_squared_error')
# clf.fit(Xtrain_t.numpy(), ytrain_pd[str(nstage-1)].values)
# bestpara = clf.best_params_

# RF_tuned_parameters =  {
#  'max_depth': [2,5,10, 50],
#  'max_features': ['auto', 'sqrt'],
#  'min_samples_leaf': [1, 2, 4],
#  'min_samples_split': [2, 5, 10],
#  'n_estimators': [5,10,50,100]}
# rf = ensemble.RandomForestRegressor()
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = RF_tuned_parameters, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)






# +
# rf_random.best_params_
# -

oldstage  = -99
allmodel = [linear_model.LinearRegression(),linear_model.ElasticNet(alpha=bestpara['alpha'], l1_ratio=bestpara['l1_ratio']),ensemble.RandomForestRegressor(**rf_random.best_params_),MLPRegressor(hidden_layer_sizes=(10, ))]
epscore = [[] for i in allmodel]
for column in ytrain_pd:
    
    nowstage = int(column[0])
    if nowstage!=oldstage:
        dfregall_pd = pd.concat(tuple([Xdataall_pd['f'][str(istage)] for istage in range(int(column[0])+1)]), axis=1)

    ytrain = ytrain_t.data.numpy()[:,i]
    ytest = ytest_t.data.numpy()[:,i]
    
    dfregtrain_pd = dfregall_pd.iloc[:Ntrain,:]
    dfregtest_pd = dfregall_pd.iloc[Ntrain:(Ntrain+Ntest),:]
    
    for imodel,model in enumerate(allmodel):
        model.fit(dfregtrain_pd.values,ytrain)
        y_pred = model.predict(dfregtest_pd.values)
        score = np.sum((ytest-y_pred)**2)/np.sum((ytest-np.mean(ytrain))**2)
        print(column,score)
        epscore[imodel].append(score)
        
    oldstage = int(column[1])
    i += 1


# +
oldstage  = -99

i = 0
for column in ytrain_pd:
    
    nowstage = int(column[0])
    if nowstage!=oldstage:
        dfregall_pd = pd.concat(tuple([Xdataall_pd['f'][str(istage)] for istage in range(int(column[0])+1)]), axis=1)

    ytrain = ytrain_t.data.numpy()[:,i]
    ytest = ytest_t.data.numpy()[:,i]
    
    dfregtrain_pd = dfregall_pd.iloc[:Ntrain,:]
    dfregtest_pd = dfregall_pd.iloc[Ntrain:(Ntrain+Ntest),:]
    
    for imodel,model in enumerate(allmodel):
        model.fit(dfregtrain_pd.values,ytrain)
        y_pred = model.predict(dfregtest_pd.values)
        score = np.sum((ytest-y_pred)**2)/np.sum((ytest-np.mean(ytrain))**2)
        print(column,score)
        epscore[imodel].append(score)
        
    oldstage = int(column[1])
    i += 1


# -

print(np.std(epscore[3]))



# +

epscore = [[] for i in allmodel]
for column in ytrain_pd:
    
    nowstage = int(column[0])
    if nowstage!=oldstage:
        dfregall_pd = pd.concat(tuple([Xdataall_pd['f'][str(istage)] for istage in range(int(column[0])+1)]), axis=1)

    ytrain = ytrain_t.data.numpy()[:,i]
    ytest = ytest_t.data.numpy()[:,i]
    
    dfregtrain_pd = dfregall_pd.iloc[:Ntrain,:]
    dfregtest_pd = dfregall_pd.iloc[Ntrain:(Ntrain+Ntest),:]
    
    for imodel,model in enumerate(allmodel):
        model.fit(dfregtrain_pd.values,ytrain)
        y_pred = model.predict(dfregtest_pd.values)
        score = np.sum((ytest-y_pred)**2)/np.sum((ytest-np.mean(ytrain))**2)
        print(column,score)
        epscore[imodel].append(score)
        
    oldstage = int(column[1])
    i += 1
# -




