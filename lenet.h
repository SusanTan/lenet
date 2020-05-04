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


float*** mnist_train_imgs;
uint8_t* mnist_train_labels;
float*** mnist_test_imgs;
uint8_t* mnist_test_labels;


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
LeNet cumulator;
/// intermediate results///////
float*** S2_max_map; // record max locations
float*** S4_max_map; // record max locations
float*** last_error; //(batchsize, 10)

float*** out_C1;
float*** out_S2;
float*** out_C3;
float*** out_S4;
float*** out_C5;
float*** out_F6;
float*** out_OL;

float*** error_C1;
float*** error_S2;
float*** error_C3;
float*** error_S4;
float*** error_C5;
float*** error_F6;
float*** error_OL;

//Intermediate out;
//Intermediate error;
///////////////////////////////

//initializations
void kaiming_uniform(float***** kernel, int in_channels, int out_channels, int kernel_size);
void uniform_W(float*** W, int in, int out);
void uniform_B(float** B, int in, int out);

//free functions
void free_conv(float**** kernel, int in_channels, int out_channels, int kernel_size);
void free_linear(float** W, float* B, int out_channels);

//forward pass
void conv2d_forward(float***** conv, float**** x, int img_size, int in_channels, int out_channels, float**** output);
void maxpool2d_forward(int stride, int pool_size, float**** in, int channels, int img_size, float**** output, float**** max_map);
void linear_forward(float*** W, float** B, float**** in, int in_channels, int out_channels, float**** output);

//backward pass
void last_layer_prep(uint8_t* label, float**** out, int out_channels, float**** delta);
void linear_backward(float**** error_l_plus_1, float**** in, float*** W_l, int l_cin, int l_cout, float**** error_l, float***W_l_delta, float** B_l_delta);
void conv_backward(float**** error_l_plus_1, float**** in, float***** W_l, int l_cin, int l_cout, float**** error_l, int kernel_size, int img_size_in, int img_size_out, float***** W_l_delta);
void pool_backward(float**** error_l_plus_1, int channels, float**** error_l, int stride, int img_size_in, float**** max_map);
