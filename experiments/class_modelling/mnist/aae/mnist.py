import numpy as np
import torch
import torchvision


transforms = transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32, 2),
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Lambda(lambda x: (x - 0.5) / 0.5)
])


def next_batch(xs, batch_size):
    idx = np.random.randint(0, xs.shape[0], batch_size)
    x_batch = xs[idx]
    return x_batch


def mnist_n(n, train, batch_size=128, data_dir='mnist/'):
    data = torchvision.datasets.MNIST(data_dir, train=train, transform=transforms)
    
    if n is not None:
        if train:
            target_idxs = np.argwhere(data.train_labels == n).flatten()
            data.train_data =  data.train_data[target_idxs]
            data.train_labels = data.train_labels[target_idxs]
            data_size = data.train_data.shape[0]
            train_data = next(enumerate(
                    torch.utils.data.DataLoader(
                            data, batch_size=data.train_data.shape[0], shuffle=False)))
            return train_data[1][0], data_size
        else:
            target_idxs = np.argwhere(data.test_labels == n).flatten()
            data.test_data =  data.test_data[target_idxs]
            data.test_labels = data.test_labels[target_idxs]
            data_size = data.test_data.shape[0]
            test_data = next(enumerate(
                    torch.utils.data.DataLoader(
                            data, batch_size=data.test_data.shape[0], shuffle=False)))
            return test_data[1][0], data_size
    else:
        data_size = data.test_data.shape[0]
        test_data = next(enumerate(
                torch.utils.data.DataLoader(
                        data, batch_size=data.test_data.shape[0], shuffle=False)))
        return test_data[1][0], data_size
    
    #mnist = enumerate(torch.utils.data.DataLoader(
    #        data, batch_size=batch_size, shuffle=True
    #))
    #return mnist, data_size # data, 
