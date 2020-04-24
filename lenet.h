#include "mnist.h"
#include "conv2d.h"
#include "MaxPool2d.h"
#include "Linear.h"

#define CONV_SIZE   5
#define POOL_SIZE   2
#define POOL_STRIDE 2
#define BATCHSIZE   32

//Channel size
#define C1_CIN      1
#define C1_COUT     6
#define C3_COUT     16
#define C5_COUT     120
#define F6_COUT     84
#define OL_COUT     10

//image size
#define C1_INSIZE   32
#define S2_INSIZE   28
#define C3_INSIZE   14
#define S4_INSIZE   10
#define C5_INSIZE   5
#define F6_INSIZE   1

Img mnist_train_imgs[60000];
uint8_t mnist_train_labels[60000];

Img** img_batch;
//kernels of size (out_channels, in_channels, h, w)
typedef struct LeNet
{
    float**** C1;
    int pool_stride;
    int pool_size;
    float**** C3;
    float**** C5;
    float** F6_W;
    float*  F6_B;
    float** OL_W;
    float*  OL_B;
}LeNet;

LeNet lenet;


Img** conv2d_forward(float**** conv, Img** img_batch, int batchsize, int img_size, int in_channels, int out_channels);
Img** maxpool2d_forward(int stride, int pool_size, Img** in, int batchsize, int channels, int img_size);
Img** linear_forward(float** W, float* B, Img** in, int batchsize, int in_channels, int out_channels);
