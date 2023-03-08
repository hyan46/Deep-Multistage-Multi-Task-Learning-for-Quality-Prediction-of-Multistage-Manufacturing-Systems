from skorch.net import NeuralNet
import torch
class Net(NeuralNet):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__(*args, **kwargs)

    def get_loss(self, y_pred, y_true, **kwargs):
        loss = torch.nn.MSELoss()(y_pred, y_true)
        loss_penalty = 0
        loss_2_penalty = 0
        for name, weight in self.module_.named_parameters():
            if "weight_ih" in name:
                loss_penalty += self.module_.l1weight * weight.pow(2).sum(0).sqrt().sum()
                loss_2_penalty += self.module_.l2weight * weight.pow(2).sum()
        loss += loss_penalty
        loss += loss_2_penalty

        self.history.record('penalty', loss_penalty)
        self.history.record('l2', loss_2_penalty)

        return loss

    def train_step(self, X, y):
        self.module_.train()
        if self.device == 'cuda':
            self.module_ = self.module_.cuda()  #

        # torch.nn.utils.clip_grad_norm_(self.module_.parameters(), self.clip)
        self.optimizer_.zero_grad()
        y_pred = self.module_(X)
        loss = self.get_loss(y_pred, y)
        loss.backward()
        self.optimizer_.step()
        return {'loss': loss, 'y_pred': y_pred}

