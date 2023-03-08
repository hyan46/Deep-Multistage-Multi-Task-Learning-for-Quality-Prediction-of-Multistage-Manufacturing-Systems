from src.model import OneStageHidden, MMS
from src.net import Net
import torch.nn as nn
import pandas as pd


#
nstage = 10
ninput = 80
otherinput = 20

noutput = 5
input_size = [ninput for i in range(nstage)]
output_size =[noutput for i in range(nstage)]
hidden_size = 10
nonactivate = None
noiselevel = 0.5
Ntrain = 500
Ntest = 2000
N = Ntrain+Ntest


from skorch.callbacks import EpochScoring
R2 = EpochScoring(scoring='explained_variance', lower_is_better=False)

import numpy as np
import pandas as pd
import torch

level1f = ['f' for i in range(nstage*ninput)]
level2f = [str(i) for i in range(nstage)  for j in range(ninput)]
# #

Xdata_pd = pd.DataFrame(data = np.random.randn(N,nstage*ninput), columns =  [level1f,level2f])
index = Xdata_pd.columns
Xdata_t = torch.from_numpy(Xdata_pd.values).float()
MMStrue =  MMS(nstage, 0, input_size, hidden_size, output_size, Xdata_pd.columns, usecuda = False, nonlinearity = nonactivate, indi=False)

ytrue_t = MMStrue.forward(Xdata_t)

#%%

level1f = ['f' for i in range(nstage*(ninput+otherinput))]
level2f = [str(i) for i in range(nstage)  for j in range(ninput+otherinput)]

Xall = np.concatenate(tuple([np.concatenate(((Xdata_pd['f'][str(istage)].values),np.random.randn(N,otherinput)), axis=1) for istage in range(nstage)]),axis=1)
Xdataall_pd = pd.DataFrame(data = Xall, columns =  [level1f,level2f])
Xdataall_t = torch.from_numpy(Xdataall_pd.values).float()

Xtrain_t = Xdataall_t[:Ntrain,:]
ytrain_t = ytrue_t[:Ntrain]
ytrain_t = torch.Tensor(ytrain_t.data).float() + noiselevel * torch.randn(ytrain_t.shape)
ytrain_pd = pd.DataFrame(data=ytrain_t.numpy(), columns = MMStrue.foward_generate_columns())

Xtest_t = Xdataall_t[Ntrain:(Ntrain+Ntest),:]
ytest_t = ytrue_t[Ntrain:(Ntrain+Ntest)]

#%%
from skorch.callbacks.lr_scheduler import WarmRestartLR, LRScheduler, CyclicLR

lr_policy = LRScheduler(policy='ExponentialLR', gamma=0.999)

l1_weight = 0.005
l2_weight = 0.00001

input_size_all = [ninput + otherinput for i in range(nstage)]
net = Net(
    lr=0.01,
    module=MMS,
    max_epochs=1000,
    batch_size=128,
    module__nstage=nstage,
    module__global_size=0,
    module__l1weight=l1_weight,
    module__l2weight=l2_weight,
    module__input_size=input_size_all,
    module__hidden_size=hidden_size,
    module__output_size=output_size,
    module__nonlinearity=nonactivate,
    module__includeoutput=False,
    module__indi=False,
    device='cuda',
    module__index=Xdataall_pd.columns,
    callbacks=[R2, lr_policy],
    criterion=nn.MSELoss,
    optimizer=torch.optim.SGD,
    optimizer__momentum=0.95,
    warm_start=True,
    iterator_train__shuffle=True)


net.fit(Xtrain_t,ytrain_t)
y_test_pred = net.predict(Xtest_t)
from sklearn.metrics import explained_variance_score
explained_variance_score(ytest_t.detach().numpy(),y_test_pred)
