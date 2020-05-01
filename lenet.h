#include "mnist.h"

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
#define C1_OUTSIZE  28
#define S2_OUTSIZE  14
#define C3_OUTSIZE  10
#define S4_OUTSIZE  5
#define C5_OUTSIZE  1
#define F6_OUTSIZE  1
#define OL_OUTSIZE  1

Img mnist_train_imgs[60000];
uint8_t mnist_train_labels[60000];
Img mnist_test_imgs[10000];
uint8_t mnist_test_labels[10000];

Img** train_img_batch;
uint8_t* train_label_batch;
Img** test_img_batch;
uint8_t* test_label_batch;
//kernels of size (out_channels, in_channels, h, w)
typedef struct LeNet
{
    float**** C1;
    float**** C3;
    float**** C5;
    float** F6_W;
    float*  F6_B;
    float** OL_W;
    float*  OL_B;
}LeNet;

LeNet lenet;
LeNet delta;
/// intermediate results///////
Img* S2_max_map; // record max locations
Img* S4_max_map; // record max locations
Img* last_error; //(batchsize, 10)

typedef struct Intermediate
{
  Img* C1;
  Img* S2;
  Img* C3;
  Img* S4;
  Img* C5;
  Img* F6;
  Img* OL;
}Intermediate;

Intermediate out;
Intermediate error;
///////////////////////////////

//initializations
void kaiming_uniform(float**** kernel, int in_channels, int out_channels, int kernel_size);
void uniform_W(float** W, int in, int out);
void uniform_B(float* B, int in, int out);

//free functions
void free_conv(float**** kernel, int in_channels, int out_channels, int kernel_size);
void free_linear(float** W, float* B, int out_channels);

//forward pass
void conv2d_forward(float**** conv, Img* x, int img_size, int in_channels, int out_channels, Img* output);
void maxpool2d_forward(int stride, int pool_size, Img* in, int channels, int img_size, Img* output, Img* max_map);
void linear_forward(float** W, float* B, Img* in, int in_channels, int out_channels, Img* output);

//backward pass
void last_layer_prep(uint8_t label, Img* out, int out_channels, Img* delta);
void linear_backward(Img* error_l_plus_1, Img* in, float** W_l, float* B_l, int l_cin, int l_cout, Img* error_l, float**W_l_delta, float* B_l_delta);
void conv_backward(Img* error_l_plus_1, Img* in, float**** W_l, int l_cin, int l_cout, Img* error_l, int kernel_size, int img_size_in, int img_size_out, float**** W_l_delta);
void pool_backward(Img* error_l_plus_1, int channels, Img* error_l, int stride, int img_size_in, Img* max_map);
