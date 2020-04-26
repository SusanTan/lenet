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

void linear_forward(float** W, float* B, Img** in, int batchsize, int in_channels, int out_channels, Img** output)
{
  for(int i=0; i<batchsize; i++)
    for(int j=0; j<out_channels; j++)
      output[i][j][0][0] = 0.0f;

  for(int i=0; i<batchsize; i++)
  {
    for(int j=0; j<out_channels; j++)
    {
      for(int k=0; k<in_channels; k++)
        output[i][j][0][0] += in[i][k][0][0]*W[j][k];
      output[i][j][0][0] += B[j];
    }
  }
}

void last_layer_backward(uint8_t* label_batch, Img** out, Img** in, float** W, float* B, int batchsize, int in_channels, int out_channels, float** delta)
{
  //form y the same size as out
  for (int i=0; i<batchsize; i++)
  {
    for(int j=0; j<out_channels; j++)
    {
      if(j==label_batch[i])
      {
        delta[i][j]=1.0f;
      }
      else
        delta[i][j]=-1.0f;
    }
  }

  //delta = (y-out)*df(out) where df(x) is 1-x*x
  for(int i=0; i<batchsize; i++)
  {
    for(int j=0; j<out_channels; j++)
    {
      float x = out[i][j][0][0];
      delta[i][j] = (delta[i][j]-x)*(1-x*x);
    }
  }

  float eta = 0.1f;
  float temp = 0.0f;
  //W += eta*<outerproduct(delta, in)> over the batch
  for(int j=0; j<out_channels; j++)
  {
    for(int k=0; k<in_channels; k++)
    {
      temp = 0.0f;
      for(int i=0; i<batchsize; i++)
        temp += delta[i][j] * in[i][k][0][0];
      temp *= eta/(float)batchsize;
      W[j][k] += temp;
    }
  }

  //B += eta * delta
  for(int i=0; i<out_channels; i++)
  {
    temp = 0.0f;
    for(int j=0; j<batchsize; j++)
      temp+=delta[j][i];
    temp *= eta/(float)batchsize;
    B[i] += temp;
  }

}

void linear_backward(float** W_l_plus_1, float** delta_l_plus_1, Img** out, Img** in, float** W_l, float* B_l, int batchsize, int l_cin, int l_cout, int l_plus1_cout, float** delta_l)
{
  //initialize the delta array;
  for(int i=0; i<batchsize; i++)
    for(int j=0; j<l_cout; j++)
      delta_l[i][j] = 0.0f;

  //mmul(W_l_plus_1 * delta)
  for(int i=0; i<batchsize; i++)
  {
    for(int j=0; j<l_cout; j++)
    {
      for(int k=0; k<l_plus1_cout; k++)
        delta_l[i][j] += W_l_plus_1[k][j] * delta_l_plus_1[i][k];
      float x = out[i][j][0][0];
      delta_l[i][j] *= (1-x*x);
    }
  }

  float eta = 0.1f;
  float temp = 0.0f;
  //W -= eta*<outerproduct(delta, in)> over the batch
  for(int j=0; j<l_cout; j++)
  {
    for(int k=0; k<l_cin; k++)
    {
      temp = 0.0f;
      for(int i=0; i<batchsize; i++)
        temp += delta_l[i][j] * in[i][k][0][0];
      temp *= eta/(float)batchsize;
      printf("%.8f, ", temp);
      W_l[j][k] -= temp;
    }
  }

  //B += eta * delta
  for(int i=0; i<l_cout; i++)
  {
    temp = 0.0f;
    for(int j=0; j<batchsize; j++)
      temp+=delta_l[j][i];
    temp *= eta/(float)batchsize;
    B_l[i] -= temp;
  }

}
