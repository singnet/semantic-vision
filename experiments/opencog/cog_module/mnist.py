"""
Example of backpropagating error through pattern matcher queries.

This example shows how to
maximize probability of producing correct sum of X and Y
where X and Y are digits from mnist dataset
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from cognets import CogModule, CogModel, get_value
from cognets import InputModule, set_value

try:
    from opencog.utilities import tmp_atomspace
    from opencog.scheme_wrapper import *
    from opencog.atomspace import AtomSpace, types, PtrValue
    from opencog.atomspace import create_child_atomspace
    from opencog.type_constructors import *
    from opencog.utilities import initialize_opencog, finalize_opencog
    from opencog.bindlink import execute_atom
except RuntimeWarning as e:
    pass


class Product(CogModule):
    def forward(self, x, y):
        result =  x * y
        assert result <= 1.0
        return result


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
        return result


class ProbOfDigit(CogModule):
    def forward(self, probs, i):
        return probs[0][i]


class TorchSum(CogModule):
    def forward(self, *args):
        if len(args) == 1:
            result = sum(*args)
        else:
            result = sum(args)
        if result > 1:
            import pdb;pdb.set_trace()
        return result


class MnistModel(CogModel):
    def __init__(self, atomspace):
        super().__init__()
        self.atomspace = atomspace
        self.mnist = MnistNet(ConceptNode("mnist"))
        self.prod = Product(ConceptNode("Product"))
        self.digit_prob = ProbOfDigit(ConceptNode("ProbOfDigit"))
        self.torch_sum = TorchSum(ConceptNode("TorchSum"))
        self.inh_weights = torch.nn.Parameter(torch.Tensor([0.3] * 10))
        #  create NumberNodes
        for i in range(10):
            NumberNode(str(i)).set_value(PredicateNode("cogNet"), PtrValue(i))
            inh1 = InheritanceLink(NumberNode(str(i)), ConceptNode("range"))

    def process(self, data, label):
        """
        Accepts batch with features and labels,
        returns probability of labels
        """
        with tmp_atomspace() as atomspace:
            #  compute possible pairs of NumberNodes
            vardecl, and_link = self.get_all_pairs(label, atomspace)

            # setup input images
            inp1 = InputModule(ConceptNode("img1"), data[0].reshape([1,1, 28, 28]))
            inp2 = InputModule(ConceptNode("img2"), data[1].reshape([1,1, 28, 28]))
            return self.p_correct_answer(inp1, inp2, vardecl, and_link)


    def p_correct_answer(self, inp1, inp2, vardecl, and_link):
        """
        compute probability of earch pair
        compute total probability - sum of pairs
        """
        lst = []
        p_digit = lambda mnist, digit: self.digit_prob.execute(mnist, digit)
        pd1 = p_digit(self.mnist.execute(inp1.execute()), VariableNode("X"))
        pd2 = p_digit(self.mnist.execute(inp2.execute()), VariableNode("Y"))
        prod_expr = self.prod.execute(pd1, pd2)
        bind1 = BindLink(vardecl, and_link, prod_expr)
        prob_sum = self.torch_sum.execute(bind1)
        result = self.execute_atom(prob_sum)
        return result

    def get_all_pairs(self, label, atomspace):
        """
        Calculate all suitable pairs of digits for given label
        """
        label = str(int(label.sum()))
        var_x = atomspace.add_node(types.VariableNode, "X")
        var_y = atomspace.add_node(types.VariableNode, "Y")
        vardecl = VariableList(TypedVariableLink(var_x, TypeNode("NumberNode")), TypedVariableLink(var_y, TypeNode("NumberNode")))
        eq = EqualLink(PlusLink(var_x, var_y), NumberNode(label))
        inh1 = InheritanceLink(var_x, ConceptNode("range"))
        inh2 = InheritanceLink(var_y, ConceptNode("range"))
        return vardecl, AndLink(inh1, inh2, eq)


def train(model, device, train_loader, optimizer, epoch, log_interval, scheduler):
    model.train()
    mean = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model.zero_grad()
        output = model.process(data, target)
        loss = - torch.log(output)
        loss.backward()
        optimizer.step()
        scheduler.step()
        mean = mean * 0.99 + 0.01 * output.detach().numpy()
        if batch_idx % log_interval == 0:
            for group in optimizer.param_groups:
                lr = group['lr']
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},\t lr: '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), lr)
            print("probability P(d1,d2)={0}".format(mean))


def exponential_lr(decay_rate, global_step, decay_steps, staircase=False):
    if staircase:
        return decay_rate ** (global_step // decay_steps)
    return decay_rate ** (global_step / decay_steps)


def main():
    atomspace = AtomSpace()
    initialize_opencog(atomspace)
    device = 'cpu'
    epoch = 20
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
    for i in range(epoch):
        train(model, device, train_loader, optimizer, i + 1, 100, scheduler)
        torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
    main()
