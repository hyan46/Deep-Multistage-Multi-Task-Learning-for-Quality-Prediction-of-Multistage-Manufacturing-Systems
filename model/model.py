from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import pandas as pd
import numpy as np
from functools import reduce
import pdb

import math


class OneStageHidden(nn.Module):
    # This function use connect the hidden state and input state and return output for each stage
    
    def __init__(self, input_size, hidden_size,hidden2_size, output_size, bias=True, nonlinearity="relu", indi=True, initialize=None,  usecuda=True):
        super(OneStageHidden, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.indi = indi
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        
        self.nonlinearity = nonlinearity
        self.output_size = output_size
        self.weight_ho = Parameter(torch.Tensor(output_size, hidden_size))
        
        self.bias_ho = Parameter(torch.Tensor(output_size))
        self.usecuda = usecuda

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
            self.weight_hh = Parameter(initialize['weight_hh'])
            self.weight_ih = Parameter(initialize['weight_ih'])
            self.bias_ih = Parameter(initialize['bias_ih'])
            
        if output_size != 0:
            if hidden2_size != 0:
                self.output = nn.Sequential(
#                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_size, hidden2_size),
                    nn.ReLU(),
                    nn.Linear(hidden2_size, output_size)
                )
            else:
                self.output = nn.Sequential(
                    nn.Dropout(0.5),
 #                   nn.BatchNorm1d(hidden_size),                    
                    nn.Linear(hidden_size, output_size))
        else:
            self.output = None

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
                    nn.init.normal_(weight, 0, 1 / math.sqrt(weight.size(1)))

    def forward(self, input, hx, isoutput=True,isinput=True):
        if isinput:
            transinput = F.linear(input, self.weight_ih, self.bias_ih) 
        else:
            transinput = 0
                          
        if self.indi:
            current_hidden = transinput + F.mul(self.weight_hh, hx)
        else:
            current_hidden = transinput + F.linear(hx, self.weight_hh)

        if self.activation is not None:
            current_hidden = self.activation(current_hidden)

        if isoutput:
            output = F.linear(current_hidden, self.weight_ho, self.bias_ho)
        else:
            output = torch.Tensor([])  # Need to use this for an empty tensor
        if self.usecuda:
            output = output.cuda()
        return current_hidden, output



class MMS(nn.Module):
    def __init__(self, nstage, hidden_size, train_columns, test_continuous_columns,test_discrete_columns, hidden2_size = None, process_keyword=['f'],quality_keywords=['v'], nonlinearity=None, includeoutput=False, initialize=[None], usecuda=True, indi=True, outputstage=range(10)):
        super(MMS, self).__init__()
        self.nstage = nstage
        self.combined_input_size = np.zeros(self.nstage)
        self.hidden_size = hidden_size
        self.onestagemodel = [[] for i in range(self.nstage)]
        self.alloutputstage = outputstage
        self.includeoutput = includeoutput
        self.usecuda = usecuda
        self.process_keyword = process_keyword
        self.quality_keywords = quality_keywords
        self.train_columns = train_columns
        self.test_continuous_columns = test_continuous_columns
        self.test_discrete_columns = test_discrete_columns
        self.output_size = [0 for i in range(self.nstage)]
        self.input_size = [0 for i in range(self.nstage)]
        self.hidden = []
        self.get_size()
        
        if len(initialize) == 1:
            initialize = [initialize[0] for i in range(self.nstage)]                        
        self.stagelist = [[] for istage in range(self.nstage)]
        for istage in range(self.nstage):
#            print(self.combined_input_size[istage], self.hidden_size)
            self.stagelist[istage] = OneStageHidden(int(self.combined_input_size[istage]), self.hidden_size,hidden2_size,self.output_size[istage], bias=True, nonlinearity=nonlinearity, indi=indi, initialize=initialize[istage], usecuda=self.usecuda)
        self.onestagemodel = nn.ModuleList(self.stagelist)

    def get_size(self):
        self.input_size = [0 for i in range(self.nstage)]
        output_continuous_size = [0 for i in range(self.nstage)]
        output_discrete_size = [0 for i in range(self.nstage)]
        output_in_size = [0 for i in range(self.nstage)]
        
        for istage in range(self.nstage):
            self.input_size[istage] = self.get_stage_size(self.train_columns,self.process_keyword, istage)
            output_continuous_size[istage] = self.test_continuous_columns.get_locs(str(istage)).size
            if self.test_discrete_columns is not None:
                output_discrete_size[istage] = self.test_discrete_columns.get_locs(str(istage)).size
                
            if self.includeoutput:
                output_in_size[istage] =  int(self.get_stage_size(self.train_columns,self.quality_keywords,istage))
        self.combined_input_size[0] = self.input_size[0]
        for istage in range(1,self.nstage):
            self.combined_input_size[istage] = int(self.input_size[istage] + output_in_size[istage-1])
            
        for istage in range(self.nstage):
            self.output_size[istage] = (output_continuous_size[istage] + output_discrete_size[istage])
        return output_continuous_size,output_discrete_size, self.input_size        
    
    def get_stage_size(self, index, variable_type, stage):
        list_index = [index.get_locs((variable_type[i], str(stage))) for i in range(len(variable_type))]
        allindex = reduce(np.union1d, list_index)
        return int(allindex.size)
    
    def get_loc(self, index, variable_type, stage):
        # Input: variable_type: should be a list, can be single variable type such as ['f'] or multiple such as ['f','v'].
        # stage: should be int, such as 0, 1, 2,
        # Get location of all
        list_index = [index.get_locs((variable_type[i], str(stage))) for i in range(len(variable_type))]
        return torch.LongTensor(reduce(np.union1d, list_index))

    def foward_generate_columns(self):
        level1v = []
        level2v = []
        for i in range(self.nstage):
            level1v += ['v' for i in range(self.output_size[i])]
            level2v += [str(i) for j in range(self.output_size[i])]
        return [level1v, level2v]

    def forward(self, data):
        hidden = torch.zeros(data.shape[0], self.hidden_size)
        if self.usecuda:
            hidden = hidden.cuda()
        output_pred = [None for i in range(self.nstage)]

        for istage in range(self.nstage):
            isoutput = (self.output_size[istage] != 0)
            isinput = (self.input_size[istage] != 0)
            
            pdnow = data[:, self.get_loc(self.train_columns,self.process_keyword, istage)]
            if self.includeoutput:
                if istage > 0:
                    pdnow = torch.cat((pdnow, data[:, self.get_loc(self.train_columns,self.quality_keywords, istage - 1)]), 1)
            
            combined_input = pdnow
            hidden, output_pred[istage] = self.onestagemodel[istage](combined_input, hidden, isoutput, isinput)
            self.hidden.append(hidden)
        output_combined = torch.cat(tuple([output_pred[i] for i in self.alloutputstage]), 1)
        return output_combined

