#include <stdlib.h>
#include <math.h>
#include "mnist.h"

float RandomNumber(float Min, float Max)
{
    return ((float)rand()/(float)RAND_MAX) * (Max - Min) + Min;
}

void uniform_W(float** W, int in, int out)
{
  float k = 1/(float)in;
  float sqrtk = sqrtf(k);
  for(int i = 0; i<out; i++)
    for(int j=0; j<in; j++)
      W[i][j] = RandomNumber(-sqrtk, sqrtk);
}

void uniform_B(float* B, int in, int out)
{
  float k = 1/(float)in;
  float sqrtk = sqrtf(k);
  for(int i=0; i<out; i++)
    B[i] = RandomNumber(-sqrtk,sqrtk);
}


float** initialize_linear_weight(int in, int out)
{
  //malloc space of shape (out,in) for weight
  float** W = (float**) malloc (out*sizeof(float*));
  for(int i=0; i<out; i++)
    W[i]=(float*) malloc (in*sizeof(float));
  uniform_W(W, in, out);
  return W;
}

float* initialize_linear_bias(int in, int out)
{
  //malloc space of shape (out) for bias
  float* B = (float*) malloc (out*sizeof(float));
  uniform_B(B, in, out);
  return B;
}

Img** linear_forward(float** W, float* B, Img** in, int batchsize, int in_channels, int out_channels)
{
  Img** forward = (Img**) malloc (sizeof(Img*)*batchsize);
  for(int i=0; i<batchsize; i++)
    forward[i] = (Img*)malloc(sizeof(Img)*out_channels);

  for(int i=0; i<batchsize; i++)
  {
    for(int j=0; j<out_channels; j++)
    {
      float** out = (float**) malloc (sizeof(float*));
      out[0] = (float*) malloc (sizeof(float));
      out[0][0] = 0.0f;
      for(int k=0; k<in_channels; k++)
      {
        out[0][0] += in[i][k][0][0]*W[j][k];
      }
      out[0][0] += B[j];
      forward[i][j] = out;
    }
  }
  return forward;
}
