from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import pandas as pd
import numpy as np
from functools import reduce
import pdb
from indrnn.indrnn import IndRNN
import math
class OneStageHidden(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,  bias=True, nonlinearity="relu", indi = True, initialize = None, usecuda=True):
        super(OneStageHidden, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.indi = indi
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.nonlinearity = nonlinearity
        self.output_size = output_size

        self.weight_ho = Parameter(torch.Tensor(output_size, hidden_size))
        self.bias_ho = Parameter(torch.Tensor(output_size))
        self.usecuda=usecuda


        # Use Independent Hidden state or not
        if self.indi:
            self.weight_hh = Parameter(torch.Tensor(hidden_size))
        else:
            self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))

        # Nonlinear activation functions
        if self.nonlinearity == "tanh":
            self.activation = F.tanh
        elif self.nonlinearity == "relu":
            self.activation = F.relu
        else:
            self.activation = None

        # Include Bias or not
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)

        if initialize is None:
            self.reset_parameters()
        else:
            self.weight_ho = Parameter(initialize['weight_ho'])
            self.weight_hh = Parameter(initialize['weight_hh'])
            self.weight_ih = Parameter(initialize['weight_ih'])
            self.bias_ho = Parameter(initialize['bias_ho'])
            self.bias_ih = Parameter(initialize['bias_ih'])


    def reset_parameters(self):
        # Initialization of weights
        for name, weight in self.named_parameters():
            if weight.size()[0] == 0:
                continue
            elif len(weight.size()) == 2:
                nn.init.normal_(weight, 0, 1 / math.sqrt(weight.size(1)))
            elif len(weight.size()) == 1:
                nn.init.normal_(weight, 0, 1 / math.sqrt(weight.size(0)))
                
            if self.indi:
                if "bias_ih" in name:
                    weight.data.zero_()
                elif "weight_hh" in name:
                    nn.init.constant_(weight, 1)
                elif "weight_ih" in name:
                    nn.init.normal_(weight, 0, 0.01)
                    

            

    def forward(self, input, hx, isoutput = True):
        if self.indi:
            current_hidden = (F.linear(input, self.weight_ih, self.bias_ih) + F.mul(self.weight_hh, hx))
        else:
            current_hidden = (F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh))

        if self.activation is not None:
            current_hidden = self.activation(current_hidden)

        if isoutput:
            output = F.linear(current_hidden, self.weight_ho, self.bias_ho)
        else:
            output =  torch.Tensor([]) # Need to use this for an empty tensor
        if self.usecuda:
            output = output.cuda()
        return current_hidden,output




#%%

class MMS(nn.Module):
    def __init__(self, nstage, global_size, input_size, hidden_size, output_size, index,nlayer=1, process_keyword = ['f'], l1weight=0.0, l2weight=0.00,quality_continuous_keywords = ['v'], quality_discrete_keywords = ['R'], nonlinearity = None, includeoutput = False, initialize = [None], usecuda = True, classification_idx_start = None, indi=True,outputstage = range(10)):
        super(MMS, self).__init__()
        self.nstage = nstage
        self.combined_input_size = np.arange(self.nstage)
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.global_size = global_size
        self.input_size = input_size
        self.onestagemodel = [[] for i in range(self.nstage)]
        self.alloutputstage = outputstage
        self.includeoutput = includeoutput
        self.include_classification = classification_idx_start is None
        self.usecuda = usecuda
        self.process_keyword = process_keyword
        self.quality_continuous_keywords = quality_continuous_keywords
        self.quality_discrete_keywords = quality_discrete_keywords
        self.l1weight = l1weight
        self.nlayer = nlayer
        self.l2weight = l2weight
        if len(initialize) == 1:
            initialize = [initialize[0] for i in range(self.nstage)]

        for istage in range(self.nstage):
            self.combined_input_size[istage] = self.input_size[istage] + self.global_size
            if includeoutput:
                if istage > 0:
                    self.combined_input_size[istage] += self.output_size[istage - 1]
        self.index = index
        self.stagelist = [[] for istage in range(self.nstage)]
        self.output_stage = [None for istage in range(self.nstage)]
        for istage in range(self.nstage):
            print(istage)
            self.stagelist[istage] = IndRNN(
            self.combined_input_size[istage], self.hidden_size, n_layer=self.nlayer, batch_norm=True,
            hidden_max_abs=None, batch_first=True,
            bidirectional=False, recurrent_inits=None,
            gradient_clip=5)

        for istage in range(self.nstage):
            if istage in self.alloutputstage:
                self.output_stage[istage] = nn.Linear(self.hidden_size,self.output_size[istage])
            # OneStageHidden(self.combined_input_size[istage], self.hidden_size,self.output_size[istage], bias=True, nonlinearity= nonlinearity, indi=indi, initialize = initialize[istage],  usecuda=self.usecuda)
        self.onestagemodel =  nn.ModuleList(self.stagelist)
        self.alloutput = nn.ModuleList(self.output_stage)

    def get_loc(self,variable_type,stage):
        # Input: variable_type: should be a list, can be single variable type such as ['f'] or multiple such as ['f','v'].
        # stage: should be int, such as 0, 1, 2,
        # Get location of all
        list_index= [self.index.get_locs((variable_type[i], str(stage))) for i in range(len(variable_type))]
        return torch.LongTensor(reduce(np.union1d,list_index))

    def foward_generate_columns(self):
        level1v = []
        level2v = []
        for i in range(self.nstage):
            level1v += ['v' for i in range(self.output_size[i])]
            level2v += [str(i) for j in range(self.output_size[i])]
        return [level1v, level2v]

    def forward(self, data):
        hidden = torch.zeros(self.nlayer,data.shape[0],self.hidden_size)
        if self.usecuda:
            hidden = hidden.cuda()
        output_pred = [None for i in range(self.nstage)]
                 
        for istage in range(self.nstage):
            isoutput = (self.output_size[istage]!=0)
            pdnow = data[:,self.get_loc(self.process_keyword,istage)]
            if self.includeoutput:
                if istage>0:
                    pdnow = torch.cat((pdnow, data[:,self.get_loc(self.quality_continuous_keywords,istage-1)]),1)
#            if global_var is not None:
#                pdnow = pd.concat(global_var,pdnow, axis = 1)
            combined_input = pdnow
            output , hidden = self.onestagemodel[istage](combined_input.unsqueeze(0), hidden)
            if self.output_stage[istage] is not None:
                output_pred[istage] = self.output_stage[istage](output)[0,:,:]
            else:
                output_pred[istage] = None
        output_combined = torch.cat(tuple([output_pred[i] for i in self.alloutputstage]), 1)
        return output_combined
