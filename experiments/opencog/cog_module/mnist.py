from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from module import CogModule, get_cached_value
from module import InputModule


try:
    from opencog.scheme_wrapper import *
    from opencog.atomspace import AtomSpace, types, PtrValue, valueToPtrValue
    from opencog.atomspace import create_child_atomspace
    from opencog.type_constructors import *
    from opencog.utilities import initialize_opencog
    from opencog.bindlink import bindlink, execute_atom
except RuntimeWarning as e:
    pass


class SumProb(CogModule):
    def forward(self, x, y):
        print("SumProb")
        print(x, y)
        return x * y


class MnistNet(CogModule):
    def __init__(self, atom):
        super().__init__(atom)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        result = F.softmax(x, dim=1)
        print("forward")
        print(result)
        return result


class ProbOfDigit(CogModule):
    def forward(self, probs, i):
        print("prob of {} in {} is {}".format(i, probs, probs[0][i]))
        return probs[0][i]


class TorchSum(CogModule):
    def forward(self, *args):
        result = sum(args)
        if result > 1:
            import pdb;pdb.set_trace()
        return result


class MnistModel(nn.Module):
    def __init__(self, atomspace):
        super().__init__()
        self.atomspace = atomspace
        self.mnist = MnistNet(ConceptNode("mnist"))
        self.sum_prob = SumProb(ConceptNode("SumProb"))
        self.digit_prob = ProbOfDigit(ConceptNode("ProbOfDigit"))
        self.torch_sum = TorchSum(ConceptNode("TorchSum"))
        for i in range(10):
            NumberNode(str(i)).set_value(PredicateNode("cogNet"), PtrValue(i))
            InheritanceLink(NumberNode(str(i)), ConceptNode("range"))


    def compute_prob(self, data, label):
        """
        Accepts batch with features and labels,
        returns probability of labels
        """
        #  1) create NumberNodes
        #  2) compute possible pairs of NumberNodes
        #  3) compute probability of earch pair
        #  4) compute total probability
        atomspace = create_child_atomspace(self.atomspace)
        initialize_opencog(atomspace)
        print("compute_prob")

        inp1 = InputModule(atomspace.add_node(types.ConceptNode, "img1"), data[0].reshape([1,1, 28, 28]))
        inp2 = InputModule(atomspace.add_node(types.ConceptNode, "img2"), data[1].reshape([1,1, 28, 28]))
        pairs = bindlink(atomspace, self.get_query(str(int(label.sum())), atomspace))
        lst = []
        for pair in pairs.out:
            lst.append(self.sum_prob.execute(self.digit_prob.execute(self.mnist.execute(inp1.execute()), pair.out[0]),
                                       self.digit_prob.execute(self.mnist.execute(inp2.execute()), pair.out[1])))
        sum_query = self.torch_sum.execute(*lst)
        result = execute_atom(atomspace, sum_query)
        torch_value = get_cached_value(result)
        return torch_value

    def get_query(self, label, atomspace):
        var_x = atomspace.add_node(types.VariableNode, "X")
        var_y = atomspace.add_node(types.VariableNode, "Y")
        vardecl = VariableList(TypedVariableLink(var_x, TypeNode("NumberNode")), TypedVariableLink(var_y, TypeNode("NumberNode")))
        eq = EqualLink(PlusLink(var_x, var_y), NumberNode(label))
        inh1 = InheritanceLink(var_x, ConceptNode("range"))
        inh2 = InheritanceLink(var_y, ConceptNode("range"))
        g2 = BindLink(vardecl, AndLink(inh1, inh2, eq), ListLink(var_x, var_y))
        return g2


def train(model, device, train_loader, optimizer, epoch, log_interval, scheduler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model.zero_grad()
        output = model.compute_prob(data, target)
        loss = - torch.log(output)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_idx % log_interval == 0:
            for group in optimizer.param_groups:
                lr = group['lr']
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},\t lr: '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), output.item()), lr)


def exponential_lr(decay_rate, global_step, decay_steps, staircase=False):
    if staircase:
        return decay_rate ** (global_step // decay_steps)
    return decay_rate ** (global_step / decay_steps)


def main():
    atomspace = AtomSpace()
    initialize_opencog(atomspace)
    device = 'cpu'
    epoch = 200
    batch_size = 2
    lr = 0.0001
    decay_rate = 0.9
    decay_steps = 10000
    train_loader = torch.utils.data.DataLoader(
       datasets.MNIST('/tmp/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=batch_size, shuffle=True)
    model = MnistModel(atomspace).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    l = lambda step: exponential_lr(decay_rate, step, decay_steps,staircase=True)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l)
    train(model, device, train_loader, optimizer, epoch, 200, scheduler)

if __name__ == '__main__':
    main()
