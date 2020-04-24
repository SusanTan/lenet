#include "mnist.h"
#include "conv2d.h"
#include "MaxPool2d.h"
#include "Linear.h"

typedef struct LeNet
{
    Conv2d C1;
    MaxPool2d S2;
    Conv2d C3;
    MaxPool2d S4;
    Conv2d C5;
    Linear F6;
    Linear OL;
}LeNet;

LeNet lenet;


Imgs conv2d_forward(Conv2d* conv, Imgs x, int batchsize, int img_size);
Imgs maxpool2d_forward(MaxPool2d layer, Imgs in, int batchsize, int channels, int img_size);
Imgs linear_forward(Linear layer, Imgs in, int batchsize, int in_channels, int out_channels);
