#include <stdlib.h>
#include <math.h>
#include "mnist.h"

float RandomNumber(float Min, float Max)
{
    return ((float)rand()/(float)RAND_MAX) * (Max - Min) + Min;
}

void uniform_W(float*** W, int in, int out)
{
  float k = 1.0f/(float)(in);
  float sqrtk = sqrtf(k);
  for(int i = 0; i<out; i++)
    for(int j=0; j<in; j++)
      (*W)[i][j] = RandomNumber(-sqrtk, sqrtk);
}

void uniform_B(float** B, int in, int out)
{
  float k = 3.0f/(float)(in);
  float sqrtk = sqrtf(k);
  for(int i=0; i<out; i++)
    (*B)[i] = RandomNumber(-sqrtk,sqrtk);
}

void free_linear(float** W, float* B, int out_channels)
{
  for(int i=0; i<out_channels; i++)
    free(W[i]);
  free(W);
  free(B);
}

void linear_forward(float*** W, float** B, float**** in, int in_channels, int out_channels, float**** output)
{
  for(int i=0; i<out_channels; i++)
    (*output)[i][0][0] = 0.0f;

  for(int j=0; j<out_channels; j++)
  {
    for(int k=0; k<in_channels; k++)
      (*output)[j][0][0] += (*in)[k][0][0]*(*W)[j][k];
    (*output)[j][0][0] += (*B)[j];
  }
}

void last_layer_prep(uint8_t* label, float**** out, int out_channels, float**** error)
{
  //form y the same size as out
  for(int j=0; j<out_channels; j++)
  {
    if(j==(*label))
      (*error)[j][0][0]=1.0f;
    else
      (*error)[j][0][0]=-1.0f;
  }

  //delta = -(y-out)*df(out) where df(x) is 1-x*x
  for(int j=0; j<out_channels; j++)
  {
    float x = (*out)[j][0][0];
    (*error)[j][0][0] = ((*error)[j][0][0]-x)*(1-x*x);
  }
}

void linear_backward(float**** error_l_plus_1, float**** in, float*** W_l, int l_cin, int l_cout, float**** error_l, float*** W_l_delta, float** B_l_delta)
{
  //initialize the error array;
  for(int i=0; i<l_cin; i++)
    (*error_l)[i][0][0] = 0.0f;

  //mmul(W * delta)
  for(int j=0; j<l_cin; j++)
  {
    for(int k=0; k<l_cout; k++)
      (*error_l)[j][0][0] += (*W_l)[k][j] * (*error_l_plus_1)[k][0][0];
    float x = (*in)[j][0][0];
    (*error_l)[j][0][0] *= (1-x*x);
  }

  //float eta = 0.1f;
  for(int j=0; j<l_cout; j++)
    for(int k=0; k<l_cin; k++)
      (*W_l_delta)[j][k] += (*error_l_plus_1)[j][0][0] * (*in)[k][0][0];

  for(int i=0; i<l_cout; i++)
    (*B_l_delta)[i] += (*error_l_plus_1)[i][0][0];
}
