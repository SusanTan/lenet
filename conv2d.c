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

void kaiming_uniform(Conv2d* conv){

    //fan-in fan-out calculation based on pytorch documentation
    int num_input_fmaps = conv->in_channels;
    printf("number of input fmaps, %d\n", num_input_fmaps);
    int num_output_fmaps = conv->out_channels;
    printf("number of output fmaps, %d\n", num_output_fmaps);
    int receptive_field_size = conv->width*conv->height;

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
        for(int k=0; k<conv->width; k++)
          for(int l=0; l<conv->height; l++)
            conv->kernel[i][j][k][l] = RandomFloat(-bound,bound);

}

void initialize_conv(Conv2d* conv, int in_channels, int out_channels,
    int kernel_width, int kernel_height)
{
  conv->in_channels = in_channels;
  conv->out_channels = out_channels;
  conv->width = kernel_width;
  conv->height = kernel_height;

  //malloc space for kernel of shape (out, in, width, height)
  conv->kernel = (Kernel**) malloc (sizeof(Kernel*)*out_channels);
  for (int i=0; i<out_channels; i++)
  {
    conv->kernel[i] = (Kernel*) malloc (sizeof(Kernel)*in_channels);
    for(int j=0; j<in_channels; j++)
    {
      conv->kernel[i][j] = (float**) malloc (sizeof(float*)*kernel_width);
      for(int k=0; k<kernel_width; k++)
      {
        conv->kernel[i][j][k] = (float*) malloc (sizeof(float)*kernel_height);
      }
    }
  }

  kaiming_uniform(conv);
}


void convolute_and_cumulate (float** result, Kernel kernel, mnist_data input, int h, int w)
{
  int right = h-4;
  int down = w-4;

  for(int i=0; i<right; i++)
  {
    for(int j=0; j<down; j++)
    {

      float muladd = 0.0f;
      //compute the convolution part;
      for(int k=0; k<5; k++)
        for(int l=0; l<5; l++)
          muladd += kernel[k][l] * input.image[i+k][j+l];

      result[i][j] += muladd;
    }
  }

}


Imgs conv2d_forward(Conv2d* conv, Imgs x, int batchsize, int img_size)
{
  int out_h = img_size - 4;//assuming 5x5 kernel stride 1
  int out_w = img_size - 4;// assuming all images in the deck is the same size

  //maloc the final forward result
  Imgs forward = (mnist_data**)malloc(sizeof(mnist_data*)*batchsize);
  for(int i=0; i<batchsize; i++)
    forward[i] = (mnist_data*)malloc(sizeof(mnist_data)*conv->out_channels);


  for(int i=0; i<batchsize; i++)
  {
    for(int j=0; j<conv->out_channels; j++)
    {

      float** result = (float**) malloc(sizeof(float*)*out_h);
      for(int m=0; m<out_h; m++)
         result[m] = (float*) malloc (sizeof(float)*out_w);

      for(int m=0; m<out_h; m++)
        for(int n=0; n<out_w; n++)
          result[m][n] = 0.0f;

      for(int k=0; k<conv->in_channels; k++)
        convolute_and_cumulate(result, conv->kernel[j][k], x[i][k], img_size, img_size);

      forward[i][j].label = x[i][0].label; //not sure how important this is
      forward[i][j].image = result;
    }
  }

  return forward;
}
