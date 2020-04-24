#include "MaxPool2d.h"
#include "mnist.h"
#include <stdlib.h>

void initialize_pool(MaxPool2d* pool, int width, int length, int stride)
{
  pool->stride = stride;
  pool->width  = width;
  pool->length = length;
}

Imgs maxpool2d_forward(MaxPool2d layer, Imgs in, int batchsize, int channels, int img_size)
{
  int stride = layer.stride;
  int kernel_width = layer.width;
  int kernel_length = layer.length;
  int hout = (img_size-kernel_length)/stride + 1;
  int wout = (img_size-kernel_width)/stride + 1;

  //assuming square images

  Imgs forward = (mnist_data**)malloc(sizeof(mnist_data*)*batchsize);
  for(int i=0; i<batchsize; i++)
    forward[i] = (mnist_data*)malloc(sizeof(mnist_data)*channels);

  for(int i=0; i<batchsize; i++)
  {
    for(int j=0; j<channels; j++)
    {

      float** result = (float**) malloc(sizeof(float*)*hout);
      for(int k=0; k<hout; k++)
         result[k] = (float*) malloc (sizeof(float)*wout);

      for(int k=0; k<hout; k++)
        for(int l=0; l<wout; l++)
          result[k][l] = 0.0f;

      for(int m=0; m<(img_size-kernel_length+1); m+=stride)
      {
        for(int n=0; n<(img_size-kernel_width+1); n+=stride)
        {
          int max = 0;
          for(int l=0; l<kernel_length; l++)
            for(int w=0; w<kernel_width; w++)
              if(max<in[i][j].image[m+l][n+w])
                max = in[i][j].image[m+l][n+w];
          result[m/stride][n/stride] = max;
        }
      }

      forward[i][j].label = in[i][0].label; //not sure how important this is
      forward[i][j].image = result;
    }
  }

  return forward;
  float** result = (float**) malloc(sizeof(float*)*hout);
  for(int i=0; i<hout; i++)
    result[i] = (float*) malloc (sizeof(float)*wout);
}
