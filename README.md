File Structure: 



## model: 
- model/model.py: contains two classes `OneStageHidden` and  `MMS`, More about these two classes seen later
    - `OneStageHidden`: THe module links input, output and module in each stage
    - `MMS`: A module that contains several stages.
- model/net.py:  Use [skorch](http://skorch.readthedocs.io/) to wrap a NN class, with penalization



### OneStageHidden
Usage: 
- Initialization: 
    - input_size: size of input
    - hidden_size: The size of hidden variable in the stage
    - hidden2_size: If not 0, a two layer NN is used to connect the `hidden_size` hidden variable through a intermediate layer with size `hidden2_size`. 
    - bias: default to true, the bias of input to hidden
    - nonlinearity: choice of activation function, can use `None` for No nonlinear activation, `tanh` or `relu`
    - indi: If true, use independent hidden variable transitions as stated in IndRNN, if not, use the traditional matrix vector multiplication
    - usecuda: Use cuda or not

- def forward(self, input, hx, isoutput=True,isinput=True):
    - input: input of each module 
    - hx: hidden variable of the previous stage

### MMS

Usage: 
- Initialization: def __init__(self, nstage, hidden_size, train_columns, test_continuous_columns,test_discrete_columns, hidden2_size = None, process_keyword=['f'],quality_keywords=['v'], nonlinearity=None, includeoutput=False, initialize=[None], usecuda=True, indi=True, outputstage=range(10)):
    - nstage: number of stages
    - hidden_size: scalar, numbe of hidden stages
    - train_columns: pandas column MultiIndex,
        - the 1st level is the 'f' (or `process_keyword`) or 'v' (or `quality_keywords`).
        - The 2nd level is stage, defined by str
        - The 3rd level is individual sensor name 
    - test_continuous_columns: output pandas column name, Should not use 
        - The 1st level is stage
        - The 2nd level is individual sensor name

    - test_discrete_columns: output pandas column name for discrete variables, currently not used
    - hidden2_size: Same as `OneStageHidden`, default to 0 
    - nonlinearity: Same as `OneStageHidden`, default to None (No nonlinear activation)
    - includeoutput: default to False, if true, use intermediate quality variables from previous stages as output as well
    - indi: Same as `OneStageHidden`
    - The stage that has output




## For Simulation Study: 

Evaluation used in Simulation study are: 

- Simulation data generated from `gensimulation.py` are `simu_1.pkl`, `simu_2.pkl`, `simu_3.pkl`
- Simulation-Benchmarks.ipynb and Simulation-Benchmarks.py: Benchmark methods for simulation study
- Simulation-Benchmarks-MultitaskElasticNet.ipynb: Multi-task elasticnet methods
- Simulation-Proposed.py: Proposed methods
- gensimulation.py: Generate simulation data
    - Case 1. One Unified MMS: 10 stages Linear transition matrix:
    - Case 2. Three Parallel Sensor Groups in one MMS: 9 sensors are in 3 groups, from 3 independent lines (transition matrix)
    - Case 3. Three Manufacturing Lines in one MMS, Stage 1, 4, 7 are from line1, Stage 2,5,8 from Line2, Stage 3,6,9 from Line3

Evaluation: 
RMSE: mean of squared error (RMSE), defined as
```math
\sum_{k}\sum_{j}|y_{k,j}^{te}-\hat{y}_{k,j}^{te}|^{2}/|\hat{y}_{k,j}^{te}-\bar{y}_{k,j}^{tr}|^{2}
```
---

                                 Case 1               Case 2              Case 3
          Proposed            0.090 (0.037)       0.138 (0. 060)       0.134 (0.057)

    Multi-task Elastic Net    0.239 (0.134)       0.192 (0.108)        0.166 (0.072)

     Linear Regression        0.577 (0.632)       0.666 (1.173)        1.213 (1.653)

        Elastic Net           0.273 (0.162)       0.150 (0.090)        0.273 (0.127)

       Random Forest          0.863 (0.079)       0.768 (0.123)        0.822 (0.085)

         Neural Net           0.728 (0.266)       0.808 (0.330)        1.006 (0.413)
---
Identified sensors: Identified impotant sensors are not important




## To Do and Known problems
- The stage definition is by string currently, it seems not supporting stage larger than 10
- Classification models have not been added and it only support regression
- Current `MMS` use `train_columns` and `test_continuous_columns` to build input and output for each stage online for each batch, considering save the data beforehand.





