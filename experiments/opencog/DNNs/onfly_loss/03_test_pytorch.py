import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
import time

Nnets=10
input_size = 2048
learning_rate = 1e-1
Nit = 1000
Nbatch = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# we create N networks and N mask
# the target for the network will be to sum inputs at position defined by masks
                            
def create_networks():
    nets = []  
    for i in range(Nnets):
        model = nn.Sequential(
        nn.Linear(2048, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
        ).to(device)
        index = i % input_size
        nets.append((model,index))
    return nets

def create_batch(Nbatch, index):
    y = np.random.randint(2, size=Nbatch)
    x = np.zeros((Nbatch,input_size))
    
    # put true to true
    x[:,index] += y
    
    # add contamination
    x += np.random.binomial(1, np.random.rand() * 0.02, size=(Nbatch,input_size))
    
    # put false to false
    x[:,index] *= y
    
    
    x[x>0.5] = 1
    return x,y
    

nets = create_networks()

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)

for it in range(Nit):
    
    sum_loss  = 0
    sum_score = 0
    
    start = time.time()
    for i in range(Nnets):
        model,sel = nets[i]
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
        inputs,labels = create_batch(Nbatch, sel)        
        inputs = Variable( torch.Tensor(inputs)).to(device)
        labels = Variable( torch.Tensor(labels)).to(device)
        logits = model(inputs).view(-1)
        predict= F.sigmoid(logits) 
        loss   = F.binary_cross_entropy(predict, labels)
        
        predict = torch.round(predict)
        score   = torch.sum(predict == labels).data.cpu().numpy()/Nbatch
        sum_score += score 
        sum_loss  += loss.data.cpu().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    t = time.time() - start
    print (it, sum_loss/Nnets, sum_score/Nnets, t)

