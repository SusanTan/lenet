#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mnist.h"

float multiply_and_add (float** m1, float** m2);

float RandomFloat(float Min, float Max)
{
    return ((float)rand()/(float)RAND_MAX) * (Max - Min) + Min;
}

void kaiming_uniform(float**** kernel, int in_channels, int out_channels, int kernel_size){

    //fan-in fan-out calculation based on pytorch documentation
    int num_input_fmaps = in_channels;
    printf("number of input fmaps, %d\n", num_input_fmaps);
    int num_output_fmaps = out_channels;
    printf("number of output fmaps, %d\n", num_output_fmaps);
    int receptive_field_size = kernel_size*kernel_size;

    int fan_in = num_input_fmaps * receptive_field_size;
    //int fan_out = num_output_fmaps * receptive_field_size;

    //calculate gain
    double a = sqrt(5.0);
    double gain = sqrt(2.0 / (1 + a*a));
    //double gain = 5.0 / 3;

    //calculate std,bound
    double std = gain / sqrt(fan_in);
    float bound = (float) sqrt(3.0) * (float) std;

    for(int i=0; i<num_output_fmaps; i++)
      for(int j=0; j<num_input_fmaps; j++)
        for(int k=0; k<kernel_size; k++)
          for(int l=0; l<kernel_size; l++)
            kernel[i][j][k][l] = RandomFloat(-bound,bound);

}

float**** initialize_conv(int in_channels, int out_channels, int kernel_size)
{
  //malloc space for kernel of shape (out, in, width, height)
  float**** kernel = (float****) malloc (sizeof(float***)*out_channels);
  for (int i=0; i<out_channels; i++)
  {
    kernel[i] = (float***) malloc (sizeof(float**)*in_channels);
    for(int j=0; j<in_channels; j++)
    {
      kernel[i][j] = (float**) malloc (sizeof(float*)*kernel_size);
      for(int k=0; k<kernel_size; k++)
        kernel[i][j][k] = (float*) malloc (sizeof(float)*kernel_size);
    }
  }
  kaiming_uniform(kernel, in_channels, out_channels, kernel_size);
  return kernel;
}


void convolute_valid (float** result, float** kernel, Img input, int kernel_size, int img_size)
{
  for(int i=0; i<img_size; i++)
    for(int j=0; j<img_size; j++)
      for(int k=0; k<kernel_size; k++)
        for(int l=0; l<kernel_size; l++)
          result[i][j] += kernel[k][l] * input[i+k][j+l];
}

void convolute_full (float** result, float** kernel, Img input, int kernel_size, int img_size)
{
  for(int i=0; i<img_size; i++)
    for(int j=0; j<img_size; j++)
      for(int k=0; k<kernel_size; k++)
        for(int l=0; l<kernel_size; l++)
          result[i+k][j+l] += kernel[k][l] * input[i][j];
}


void conv2d_forward(float**** conv, Img* x, int img_size, int in_channels, int out_channels, Img* output)
{
  //clear output first
  for(int i=0; i<out_channels; i++)
    for(int j=0; j<img_size; j++)
      for(int k=0; k<img_size; k++)
        output[i][j][k] = 0.0f;

   for(int j=0; j<out_channels; j++)
     for(int k=0; k<in_channels; k++)
       convolute_valid(output[j], conv[j][k], x[k], 5, img_size);
}

void conv_backward(Img* error_l_plus_1, Img* in, float**** W_l, int l_cin, int l_cout, Img* error_l, int kernel_size, int img_size_in, int img_size_out, float**** W_l_delta)
{
  //clear the delta array
  for(int j=0; j<l_cin; j++)
    for(int k=0; k<img_size_in; k++)
      for(int l=0; l<img_size_in; l++)
        error_l[j][k][l] = 0.0f;

  for(int j=0; j<l_cout; j++)
    for(int k=0; k<l_cin; k++)
      convolute_full(error_l[k], W_l[j][k], error_l_plus_1[j], kernel_size, img_size_out);

  for(int j=0; j<l_cin; j++)
    for(int k=0; k<img_size_in; k++)
      for(int l=0; l<img_size_in; l++)
        error_l[j][k][l] *= 1-(in[j][k][l]*in[j][k][l]);

  //batched, reverted convolution in the buffer, too lazy to do anything fancy
  for(int i=0; i<l_cout; i++)
    for(int j=0; j<l_cin; j++)
      convolute_valid(W_l_delta[i][j], error_l_plus_1[i], in[j], img_size_out, kernel_size);
}
