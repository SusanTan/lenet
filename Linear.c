#include "Linear.h"
#include <stdlib.h>
#include <math.h>
#include "mnist.h"

float RandomNumber(float Min, float Max)
{
    return ((float)rand()/(float)RAND_MAX) * (Max - Min) + Min;
}

void uniform(Linear* linear, int in, int out)
{
  float k = 1/(float)in;
  float sqrtk = sqrtf(k);
  for(int i = 0; i<out; i++)
    for(int j=0; j<in; j++)
      linear->W[i][j] = RandomNumber(-sqrtk, sqrtk);

  for(int i=0; i<out; i++)
    linear->B[i] = RandomNumber(-sqrtk,sqrtk);
}


void initialize_linear(Linear* linear, int in, int out)
{
  //malloc space of shape (out,in) for weight
  linear->W = (float**) malloc (out*sizeof(float*));
  for(int i=0; i<out; i++)
    linear->W[i]=(float*) malloc (in*sizeof(float));

  //malloc space of shape (out) for bias
  linear->B = (float*) malloc (out*sizeof(float));
  uniform(linear, in, out);
}

Imgs linear_forward(Linear layer, Imgs in, int batchsize, int in_channels, int out_channels)
{
  Imgs forward = (mnist_data**) malloc (sizeof(mnist_data*)*batchsize);
  for(int i=0; i<batchsize; i++)
  {
    forward[i] = (mnist_data*) malloc (sizeof(mnist_data)*out_channels);
  }

  for(int i=0; i<batchsize; i++)
  {
    for(int j=0; j<out_channels; j++)
    {
      float** out = (float**) malloc (sizeof(float*));
      out[0] = (float*) malloc (sizeof(float));
      out[0][0] = 0.0f;
      for(int k=0; k<in_channels; k++)
      {
        out[0][0] += in[i][k].image[0][0]*layer.W[j][k];
      }
      out[0][0] += layer.B[j];
      forward[i][j].image = out;
      forward[i][j].label = 0;//Note, really isn't a label notation here. need to change code.
    }
  }
  return forward;
}
