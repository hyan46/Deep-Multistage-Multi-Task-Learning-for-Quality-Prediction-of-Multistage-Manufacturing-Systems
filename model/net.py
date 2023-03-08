from skorch.net import NeuralNet
import torch
import gc
class Net(NeuralNet):
    def __init__(self, lr, l1weight=0.0,l2weight=0.0,ol1weight=0.0,ol2weight=0.0, *args, **kwargs):
        super(Net, self).__init__(lr=lr,*args, **kwargs)
        self.lr = lr
        self.l1weight = l1weight
        self.l2weight = l2weight
        self.ol1weight = ol1weight
        self.ol2weight = ol2weight
           
    def train_step(self, X, y):
        self.module_.train()
        if self.device == 'cuda':
            self.module_ = self.module_.cuda()  #
            X = X.cuda()
            y = y.cuda()
        # torch.nn.utils.clip_grad_norm_(self.module_.parameters(), self.clip)
      
        self.optimizer_.zero_grad()
        
        y_pred = self.module_(X)
        loss = self.get_loss(y_pred, y)
        loss.backward()
        self.optimizer_.step()
        gc.collect()
        return {'loss': loss, 'y_pred': y_pred}

    def get_loss(self, y_pred, y_true, **kwargs):
        if self.device == 'cuda':
            self.module_ = self.module_.cuda()  #
            y_true = y_true.cuda()
            y_pred = y_pred.cuda()
 
        loss = torch.nn.MSELoss()(y_pred, y_true)
        gc.collect()
        i_l1_penalty = 0
        i_l2_penalty = 0
        o_l1_penalty = 0
        o_l2_penalty = 0
        self.history.record('MSE', loss)


        for name, weight in self.module_.named_parameters():
            if "weight_ih" in name:
                i_l1_penalty += self.l1weight * weight.pow(2).sum(0).sqrt().sum()
                i_l2_penalty += self.l2weight * weight.abs().sum()
            if "output" in name and "weight" in name:
                o_l1_penalty += self.ol1weight * weight.pow(2).sum(1).sqrt().sum()
                o_l2_penalty += self.ol2weight * weight.abs().sum()

        loss += i_l1_penalty
        loss += i_l2_penalty
        loss += o_l1_penalty
        loss += o_l2_penalty

        self.history.record('i_l1', i_l1_penalty)
        self.history.record('o_l1', o_l1_penalty)
        return loss
