import unittest
import numpy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from lstm_pytorch import LstmIterator, Lstm2D
from torchvision import datasets, transforms


def gen_index(i, items_per_layer=2):
    """
    Returns slice for getting elements from list
    """
    pos = i * items_per_layer
    return slice(pos, pos + 2)


def gen_image(height, width):
    img = np.random.random((height, width, 3))

    # fill polygon
    poly = numpy.stack([numpy.random.randint(height, size=5),
                             numpy.random.randint(width, size=5)]).reshape((5, 2))
    rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
    img[rr, cc, 1] = 1
    circles = numpy.random.randint(0, 2)
    for i in range(circles):
        # fill circle
        rr, cc = circle(numpy.random.randint(height),
                        numpy.random.randint(width),
                        numpy.random.randint(5, 15), # radius
                        img.shape)
        img[rr, cc, :] = (1, 1, 0)
    return img, circles


def gen_dataset_images(num):
    res = []
    res_num = []
    for i in range(num):
        image, num = gen_image(128, 128)
        res.append(image.transpose((2, 0, 1)))
        res_num.append(num)
    return numpy.stack(res), res_num


def reshape_to_dataset(train_set_x, neib_shape):
    from theano.tensor.nnet import neighbours
    imgs = T.tensor4('imgs')
    imgs.tag.test_value = train_set_x
    neibs = []
    for ch in range(train_set_x.shape[1]):
        new_shape = list(train_set_x.shape)
        new_shape[1] = 1
        neibs.append(neighbours.images2neibs(imgs[:,ch,:,:].reshape(new_shape), neib_shape=neib_shape))
    n_channels = train_set_x.shape[1]
    if n_channels == 3:
        neib = T.stack(neibs, axis=-1).flatten(2)
    elif n_channels == 1:
        neib = neibs[0]
    else:
        raise NotImplementedError("images with {0} channels".format(n_channels))
    resh = neib.reshape((train_set_x.shape[0], neib.shape[0] // train_set_x.shape[0], neib.shape[1]))
    # scan iterates over the first dimention
    f = theano.function(inputs=[imgs], outputs=resh.transpose(1, 0, 2))

    return f(train_set_x)


def reshape_to_labels(labels, num):
    result = numpy.zeros((len(labels), num))
    for i, l in enumerate(labels):
        result[i][l] = 0.9999
    return result


class TestLstmIterator(unittest.TestCase):
    def test_iterator(self):
       ar = numpy.asarray([[[[1, 2, 3, 4,],
                             [5, 6, 7, 8,],
                             [9,10, 11,12],]]])

       ar = torch.from_numpy(ar)
       it = LstmIterator(ar, 2, 2)
       expected = [[[1, 2],
                    [5, 6]], [[3, 4,],
                              [7, 8,]], [[9,10],
                                         [0, 0]], [[11,12],
                                                   [0, 0]]]
       for item, ex in zip(it, expected):
           self.assertTrue(all((item == torch.from_numpy(numpy.asarray(ex))).flatten()))


def count_circles():
    """
    simple test - slide
    over the picture and count circles
    """
    # hyperparams
    image_shape = numpy.array([128, 128])
    neib_shape=(4, 4 )
    input_size = numpy.prod(neib_shape) * 3
    batch_size = 30
    n_labels = 2
    num_cells = 40
    look_back_idx = image_shape / neib_shape
    # new batch
    # model
    lstm = nn.LSTM(input_size + num_cells, num_cells)
    linear = nn.Linear(num_cells, n_labels)
    # params
    params = list(lstm.parameters()) + list(linear.parameters())
    # loss
    optim = torch.optim.RMSprop(params)
    criterion = nn.BCELoss()
    out_prev = torch.zeros((1, batch_size, num_cells))
    c = torch.zeros(( 1, batch_size, num_cells))
    hidden = [out_prev, c]
    # train loop
    for j in range(1000):
        train_set_x, train_set_y = gen_dataset_images(batch_size)
        train_set_x = reshape_to_dataset(train_set_x, neib_shape)
        train_set_y = reshape_to_labels(train_set_y, n_labels)
        lstm.zero_grad()
        linear.zero_grad()
        q = queue.Queue(maxsize=look_back_idx[0])
        batch_out = []
        # loop over minibatch
        for i in range(train_set_x.shape[0]):
            if i < look_back_idx[0]:
               prev_2 = torch.zeros(1, batch_size, num_cells, dtype=torch.float)
            else:
               prev_2 = q.get()
            new_shape = (1, train_set_x.shape[1], train_set_x.shape[2])
            slice = train_set_x[i,:,:].reshape(new_shape).astype(numpy.float32)
            inp = torch.cat([torch.from_numpy(slice), prev_2], dim=2)
            out, hidden = lstm(inp, hidden)
            batch_out.append(out)
            q.put(out)
        cat = torch.nn.functional.softmax(linear(out), dim=-1)
        loss = criterion(cat.reshape(cat.shape[1:]), torch.from_numpy(train_set_y.astype(numpy.float32)))
        print("batch #{0} loss {1}".format(j, loss))
        loss.backward()
        optim.step()
        out_prev = torch.randn((1, batch_size, num_cells))
        c = torch.randn(( 1, batch_size, num_cells))
        hidden = [out_prev, c]
        # detach all parameters and states
        #detach(params)
        # reintialize optimizer
        #optim = torch.optim.RMSprop(params)
        # new batch
    import pdb;pdb.set_trace()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm1 = Lstm2D(4 * 4, 10, (4,4))
        self.lstm2 = Lstm2D(2 * 2 * 10, 20, (2,2))
        self.fc1 = nn.Linear(4 * 4 * 20, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, batch):
        out = self.lstm1(batch)
        out = self.lstm2(out)
        out = out.reshape((out.shape[0], numpy.prod(out.shape[1:])))
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def mnist():
    batch_size=300
    device = 'cpu'
    lr = 0.01
    momentum = 0.5
    epochs = 30
    test_batch_size = 1000
    train_loader = torch.utils.data.DataLoader(
           datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
            batch_size=batch_size, shuffle=True)
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True)


    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, 20)
        test(model, device, test_loader)


    torch.save(model.state_dict(),"mnist_cnn.pt")





if __name__ == '__main__':
   mnist()

