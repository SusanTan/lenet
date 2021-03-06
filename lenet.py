# -*- coding: utf-8 -*-
"""“LeNet.ipynb”的副本

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IV9mAvgcglVVoa78ZmPv8539Ebe2VIMY
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional
import torchvision
import numpy as np
import time

# get data

import urllib
try:
    # For python 2
    class AppURLopener(urllib.FancyURLopener):
        version = "Mozilla/5.0"

    urllib._urlopener = AppURLopener()
except AttributeError:
    # For python 3
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

mnist_train = torchvision.datasets.MNIST(root='data', train=True, download=True) # train data only
trainimages = mnist_train.data
trainlabels = mnist_train.targets

mnist_test = torchvision.datasets.MNIST(root='data', train=False, download=True) # train data only
testimages = mnist_test.data
testlabels = mnist_test.targets

# check training data shape
print("Training Data shape is: ", list(trainimages.size()))
print("Training Target shape is: ", list(trainlabels.size()))
print("Testing Data shape is: ", list(testimages.size()))
print("Testing Target shape is: ", list(testlabels.size()))

class LeNet5(nn.Module):

    # definition of each neural network layer
    def __init__(self):
        super(LeNet5, self).__init__()
        self.C1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.S2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        ####### Complete the defition of C3 and S4 ##########
        # C3 is a convolutional layer with 16 5x5 kernels
        # S4 is a max pooling layer with 2x2 kernel and stride 2

        self.C3 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.S4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        #####################################################

        self.C5 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.F6 = nn.Linear(120, 84)

        # output layer
        self.OL = nn.Linear(84, 10)

        # record C5 & F6 activation for visualization
        self.record = {"C5":None, "F6":None}

    # definition of the forward pass
    def forward(self, x):

        # input x are (batch, 1, 32, 32) grayscale images
        # the first convolutional layer C1 with 6 kernels size 5×5 and a stride of 1
        # output image size changes from (batch, 1, 32, 32) to (batch, 6, 28, 28)
        # then pass the feature maps to the tanh activation function
        x = torch.tanh(self.C1(x))

        # pass the feature maps to a 2x2 max pooling layer S2 
        # the output image dimension decreases halved -> (batch, 6, 14, 14)
        x = self.S2(x)

        ####### Complete the foward pass C3 > S4 > C5 ###################################
        # C2 is the second convolutional layer with 16 kernels size 5×5 and a stride of 1
        # after C2, the output image size changes from (batch, 6, 14, 14) to (batch, 16, 10, 10)
        # S3 is the second 2x2 pooling layer with strid 2
        # after S4, the output image size changes from (batch, 16, 10, 10) to (batch, 16, 5, 5)
        # C5 is the third convolutional layer with 120 kernels size 5×5 and a stride of 1
        # after C5, the output image size changes from (batch, 16, 5, 5) to (batch, 120, 1, 1)
        # C3, C5 is followed by tanh activations

        x = torch.tanh(self.C3(x))
        x = self.S4(x)
        x = torch.tanh(self.C5(x))

        ##################################################################################

        # convert (batch, 120, 1, 1) feature maps to 1d features of size (batch, 120)
        x = x.view(x.size(0), -1)

        # record the activation of C5 as a numpy array
        # .detach() declares a tensor does not need gradients
        # .numpy() convert a torch tensoer without gradient to numpy array
        # independent of the forward pass, we have to make a copy of x by method .clone()
        self.record["C5"] = x.clone().detach().numpy()

        # pass the activation to the fully connected layer F6 followed by a tanh activation
        # output size changes from (batch, 120) to (batch, 86)
        x = torch.tanh(self.F6(x))

        # record the activat of F6 as numpy array
        self.record["F6"] = x.clone().detach().numpy()

        # pass the activation to the final fully connected layer OL followed by a tanh activation
        # output size changes from (batch, 86) to (batch, 10)
        x = torch.tanh(self.OL(x))

        return x

ntrain = trainimages.shape[0];  # number of training examples
ntest  = testimages.shape[0];  # number of testing examples
nepoch = 10;                    # number of epochs through training set
batchsize = 32                  # minibatch size

errs = []
losses = []

#lenet5 = LeNet5()
lenet5 = LeNet5()

# use SGD optimizer, set learning rate parameter as 0.1
optimizer = optim.SGD(lenet5.parameters(), lr=0.1)

for t in range(int(ntrain / batchsize)):
    batchindices = np.random.choice(ntrain, batchsize, replace=False)
    trainlabels_iter = trainlabels[batchindices]

    # label 1 for the correct digit and -1 for the incorrect digits
    y = torch.ones(10, batchsize) * (-1)
    y[trainlabels_iter, torch.arange(batchsize, dtype=torch.int64)] = 1

    # normalize input images
    imgs = torch.zeros([batchsize, 1, 32, 32])
    imgs[:, 0, 2: -2, 2: -2] = trainimages[batchindices].float() / 255.

    # before the forward pass, clean the gradient buffers of all parameters
    optimizer.zero_grad()

    # forward pass
    out = lenet5(imgs)

    # MSE loss
    loss = torch.mean(0.5*(y - out.t())**2)

    # backward pass
    loss.backward()

    # update parameters using SGD
    optimizer.step()

batchsize=1
for t in range(int(ntest)):
    batchindices = np.random.choice(ntest, batchsize, replace=False)
    testlabels_iter = testlabels[batchindices]

    # label 1 for the correct digit and -1 for the incorrect digits
    y = torch.ones(10, batchsize) * (-1)
    y[testlabels_iter, torch.arange(batchsize, dtype=torch.int64)] = 1

    # normalize input images
    imgs = torch.zeros([batchsize, 1, 32, 32])
    imgs[:, 0, 2: -2, 2: -2] = testimages[batchindices].float() / 255.

    # before the forward pass, clean the gradient buffers of all parameters
    optimizer.zero_grad()

    # forward pass
    #out = lenet5(imgs)
    out = lenet5(imgs)

    # MSE loss
    loss = torch.mean(0.5*(y - out.t())**2)

    # calculate error rate and loss for plot
    pred = torch.argmax(out, dim=1)
    err = torch.mean((pred != testlabels_iter).float())
    errs.append(err.detach().numpy())
    losses.append(loss.detach().numpy())
print("classification error: %f" % errs[-1])
print("loss: %f" % losses[-1])
