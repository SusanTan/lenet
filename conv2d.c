#include "conv2d.h"
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

    //calculate std,bound
    double std = gain / sqrt(fan_in);
    float bound = (float) sqrt(3.0) * std;

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
      {
        kernel[i][j][k] = (float*) malloc (sizeof(float)*kernel_size);
      }
    }
  }

  kaiming_uniform(kernel, in_channels, out_channels, kernel_size);
  return kernel;
}


void convolute_and_cumulate (float** result, float** kernel, Img input, int size)
{
  int out_size = size-4;

  for(int i=0; i<out_size; i++)
  {
    for(int j=0; j<out_size; j++)
    {

      float muladd = 0.0f;
      //compute the convolution part;
      for(int k=0; k<5; k++)
        for(int l=0; l<5; l++)
          muladd += kernel[k][l] * input[i+k][j+l];

      result[i][j] += muladd;
    }
  }

}


Img** conv2d_forward(float**** conv, Img** x, int batchsize, int img_size, int in_channels, int out_channels)
{
  int out_size = img_size - 4;//assuming 5x5 kernel stride 1
  //maloc the final forward result
  Img** forward = (Img**)malloc(sizeof(Img*)*batchsize);
  for(int i=0; i<batchsize; i++)
    forward[i] = (Img*)malloc(sizeof(Img)*out_channels);

  for(int i=0; i<batchsize; i++)
  {
    for(int j=0; j<out_channels; j++)
    {

      float** result = (float**) malloc(sizeof(float*)*out_size);
      for(int m=0; m<out_size; m++)
         result[m] = (float*) malloc (sizeof(float)*out_size);

      for(int m=0; m<out_size; m++)
        for(int n=0; n<out_size; n++)
          result[m][n] = 0.0f;

      for(int k=0; k<in_channels; k++)
        convolute_and_cumulate(result, conv[j][k], x[i][k], img_size);

      forward[i][j] = result;
    }
  }

  return forward;
}
