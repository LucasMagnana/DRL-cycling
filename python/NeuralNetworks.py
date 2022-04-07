import torch
from torch import nn
from torch.autograd import Variable




class Actor(nn.Module):

    def __init__(self, size_ob, size_action, hyperParams, max_action=1, tanh=False): #for saved hyperparameters
        super(Actor, self).__init__()
        print(hyperParams)
        self.inp = nn.Linear(size_ob, hyperParams.HIDDEN_SIZE_1)
        self.int = nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2)
        self.out = nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action)
        self.max_action = max_action
        self.tanh = tanh

    def forward(self, ob):
        ob = ob.float()
        out = nn.functional.relu(self.inp(ob))
        out = nn.functional.relu(self.int(out))
        if(self.tanh):
            return torch.tanh(self.out(out)*self.max_action)
        else:
            return self.out(out)*self.max_action

        

class REINFORCE_Model(nn.Module):
    def __init__(self, size_ob, size_action, hyperParams):
        super(REINFORCE_Model, self).__init__()       
        self.inp = nn.Linear(size_ob, hyperParams.HIDDEN_SIZE_1)
        self.out = nn.Linear(hyperParams.HIDDEN_SIZE_1, size_action)
        self.sm = nn.Softmax(dim=-1)
    
    def forward(self, ob):
        ob = ob.float()
        out = nn.functional.relu(self.inp(ob))
        out = self.sm(self.out(out))
        return out




class PPO_Actor(nn.Module):
    def __init__(self, size_ob, size_action, hyperParams):
        super(PPO_Actor, self).__init__()
        self.act_inp = nn.Linear(size_ob, hyperParams.HIDDEN_SIZE_1)
        self.act_int = nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2)
        self.act_out = nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action)
        self.sm = nn.Softmax(dim=-1)
    
    def forward(self, ob):
        ob_f = ob.float()
        act_out = nn.functional.relu(self.act_inp(ob_f))
        act_out = nn.functional.relu(self.act_int(act_out))
        act_out = self.sm(self.act_out(act_out))
        return act_out


class PPO_Critic(nn.Module):
    def __init__(self, size_ob, hyperParams):
        super(PPO_Critic, self).__init__()
        self.crit_inp = nn.Linear(size_ob, hyperParams.HIDDEN_SIZE_1)
        self.crit_int = nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2)
        self.crit_out = nn.Linear(hyperParams.HIDDEN_SIZE_2, 1)
    
    def forward(self, ob):
        ob = ob.float()
        crit_out = nn.functional.relu(self.crit_inp(ob))
        crit_out = nn.functional.relu(self.crit_int(crit_out))
        crit_out = self.crit_out(crit_out)
        return crit_out



class REINFORCE_Model(nn.Module):
    def __init__(self, size_ob, size_action, hyperParams):
        super(REINFORCE_Model, self).__init__()       
        self.inp = nn.Linear(size_ob, hyperParams.HIDDEN_SIZE_1)
        self.int = nn.Linear(hyperParams.HIDDEN_SIZE_1, hyperParams.HIDDEN_SIZE_2)
        self.out = nn.Linear(hyperParams.HIDDEN_SIZE_2, size_action)
        self.sm = nn.Softmax(dim=-1)
    
    def forward(self, ob):
        ob = ob.float()
        out = nn.functional.relu(self.inp(ob))
        out = nn.functional.relu(self.int(out))
        out = self.sm(self.out(out))
        return out






