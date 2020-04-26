#include <stdio.h>
#include <stdlib.h>
#include "lenet.h"
#include "lenet_test.h"
#include <math.h>

void RandomChoices(int* batchindice, int range, int size)
{
  for(int i=0; i<size; i++)
    batchindice[i] = (rand()%(range+1));
}

float** initialize_zeroweights(int d1, int d2)
{
  float** result = (float**)malloc(d1*sizeof(float*));
  for(int i=0; i<d1; i++)
    result[i] = (float*)malloc(d2*sizeof(float));

  for(int i=0; i<d1; i++)
    for(int j=0; j<d2; j++)
      result[i][j] = 0.0f;
  return result;
}

float* initialize_zerobias(int d1)
{
  float* result = (float*)malloc(d1*sizeof(float));
  for(int i=0; i<d1; i++)
      result[i] = 0.0f;
  return result;
}

Img* initialize_images(int d1, int d2, int d3)
{
  Img* images = (Img*)malloc(sizeof(Img)*d1);
  for(int i=0; i<d1; i++)
  {
    images[i] = (float**)malloc(sizeof(float*)*d2);
    for(int j=0; j<d2; j++)
      images[i][j] = (float*)malloc(sizeof(float)*d3);
  }
  for(int i=0; i<d1; i++)
    for(int j=0; j<d2; j++)
      for(int k=0; k<d3; k++)
        images[i][j][k] = 0.0f;
  return images;
}

float**** initialize_4Dzeros(int d1, int d2, int d3, int d4)
{
  float**** images = (float****)malloc(sizeof(float***)*d1);
  for(int i=0; i<d1; i++)
  {
    images[i] = (float***)malloc(sizeof(float**)*d2);
    for(int j=0; j<d2; j++)
    {
      images[i][j] = (float**)malloc(sizeof(float*)*d3);
      for(int k=0; k<d3; k++)
        images[i][j][k] = (float*)malloc(sizeof(float)*d4);
    }
  }
  for(int i=0; i<d1; i++)
    for(int j=0; j<d2; j++)
      for(int k=0; k<d3; k++)
        for(int l=0; l<d4; l++)
          images[i][j][k][l] = 0.0f;
  return images;
}

void initialize_lenet(){
  lenet.pool_stride = POOL_STRIDE;
  lenet.pool_size   = POOL_SIZE;
  lenet.C1 = initialize_conv(C1_CIN,  C1_COUT, CONV_SIZE);
  lenet.C3 = initialize_conv(C1_COUT, C3_COUT, CONV_SIZE);
  lenet.C5 = initialize_conv(C3_COUT, C5_COUT, CONV_SIZE);
  lenet.F6_W = initialize_linear_weight(C5_COUT, F6_COUT);
  lenet.F6_B = initialize_linear_bias  (C5_COUT, F6_COUT);
  lenet.OL_W = initialize_linear_weight(F6_COUT, OL_COUT);
  lenet.OL_B = initialize_linear_bias  (F6_COUT, OL_COUT);
  //test_initialization (lenet);

  //initialize all intermediate storages
  C1_out     = initialize_images(C1_COUT, C1_OUTSIZE, C1_OUTSIZE);
  S2_max_map = initialize_images(C1_COUT, C1_OUTSIZE, C1_OUTSIZE);
  S2_out     = initialize_images(C1_COUT, S2_OUTSIZE, S2_OUTSIZE);
  C3_out     = initialize_images(C3_COUT, C3_OUTSIZE, C3_OUTSIZE);
  S4_max_map = initialize_images(C3_COUT, C3_OUTSIZE, C3_OUTSIZE);
  S4_out     = initialize_images(C3_COUT, S4_OUTSIZE, S4_OUTSIZE);
  C5_out     = initialize_images(C5_COUT, C5_OUTSIZE, C5_OUTSIZE);
  F6_out     = initialize_images(F6_COUT, F6_OUTSIZE, F6_OUTSIZE);
  OL_out     = initialize_images(OL_COUT, OL_OUTSIZE, OL_OUTSIZE);
  last_error = initialize_images(OL_COUT, OL_OUTSIZE, OL_OUTSIZE);
  OL_error   = initialize_images(F6_COUT, F6_OUTSIZE, F6_OUTSIZE);
  F6_error   = initialize_images(C5_COUT, C5_OUTSIZE, C5_OUTSIZE);
  C5_error   = initialize_images(C3_COUT, S4_OUTSIZE, S4_OUTSIZE);
  S4_error   = initialize_images(C3_COUT, C3_OUTSIZE, C3_OUTSIZE);
  C3_error   = initialize_images(C1_COUT, S2_OUTSIZE, S2_OUTSIZE);
  S2_error   = initialize_images(C1_COUT, C1_OUTSIZE, C1_OUTSIZE);
  C1_error   = initialize_images(C1_CIN , C1_INSIZE , C1_INSIZE );

  delta.OL_W = initialize_zeroweights(OL_COUT, F6_COUT);
  delta.F6_W = initialize_zeroweights(F6_COUT, C5_COUT);
  delta.OL_B = initialize_zerobias(OL_COUT);
  delta.F6_B = initialize_zerobias(F6_COUT);
  delta.C5   = initialize_4Dzeros(C5_COUT, C3_COUT, CONV_SIZE, CONV_SIZE);
  delta.C3   = initialize_4Dzeros(C3_COUT, C1_COUT, CONV_SIZE, CONV_SIZE);
  delta.C1   = initialize_4Dzeros(C1_COUT, C1_CIN , CONV_SIZE, CONV_SIZE);
}

void torch_tanh (Img* x, int img_size, int channels)
{
  for(int i=0; i<channels; i++)
    for(int j=0; j<img_size; j++)
      for(int k=0; k<img_size; k++)
        x[i][j][k] = tanh(x[i][j][k]);
}

void forward(int i)
{
  conv2d_forward(lenet.C1, img_batch[i], C1_OUTSIZE, C1_CIN, C1_COUT, C1_out);
  torch_tanh(C1_out, C1_OUTSIZE, C1_COUT);
	maxpool2d_forward(lenet.pool_stride, lenet.pool_size, C1_out, C1_COUT, S2_OUTSIZE, S2_out, S2_max_map);
  conv2d_forward(lenet.C3, S2_out, C3_OUTSIZE, C1_COUT, C3_COUT, C3_out);
  torch_tanh(C3_out, C3_OUTSIZE, C3_COUT);
  maxpool2d_forward(lenet.pool_stride, lenet.pool_size, C3_out, C3_COUT, S4_OUTSIZE, S4_out, S4_max_map);
  conv2d_forward(lenet.C5, S4_out, C5_OUTSIZE, C3_COUT, C5_COUT, C5_out);
  torch_tanh(C5_out, C5_OUTSIZE, C5_COUT);
  linear_forward(lenet.F6_W, lenet.F6_B, C5_out, C5_COUT, F6_COUT, F6_out);
  torch_tanh(F6_out, F6_OUTSIZE, F6_COUT);
  linear_forward(lenet.OL_W, lenet.OL_B, F6_out, F6_COUT, OL_COUT, OL_out);
  torch_tanh(OL_out, OL_OUTSIZE, OL_COUT);
}

/*float mse_loss(Img** output, uint8_t* label_batch)
{
  float y[BATCHSIZE][OL_COUT];
  float mse = 0.0f;

  for (int i=0; i<BATCHSIZE; i++)
  {
    for(int j=0; j<OL_COUT; j++)
    {
      if(j==label_batch[i])
        y[i][j]=1.0f;
      else
        y[i][j]=-1.0f;
    }
  }

  float temp = 0.0f;
  for(int i=0; i<BATCHSIZE; i++)
  {
    for(int j=0; j<OL_COUT; j++)
    {
      temp = 0.5f*(y[i][j]-output[i][j][0][0])*(y[i][j]-output[i][j][0][0]);
      mse += temp;
    }
  }
  mse /= BATCHSIZE*OL_COUT;
  return mse;
}*/

void backward(i)
{
  //float loss = mse_loss(output, label_batch);
  last_layer_prep  (label_batch[i], OL_out, OL_COUT, last_error);
  linear_backward  (last_error, F6_out, lenet.OL_W, lenet.OL_B,
                    F6_COUT, OL_COUT, OL_error, delta.OL_W, delta.OL_B);
  linear_backward  (OL_error, C5_out, lenet.F6_W, lenet.F6_B,
                    C5_COUT, F6_COUT, F6_error, delta.F6_W, delta.F6_B);
  conv_backward    (F6_error, S4_out, lenet.C5, C3_COUT, C5_COUT,
                    C5_error, CONV_SIZE, S4_OUTSIZE, C5_OUTSIZE, delta.C5);
  pool_backward    (C5_error, C3_COUT, S4_error, POOL_STRIDE,
                    C3_OUTSIZE, S4_max_map);
  conv_backward    (S4_error, S2_out, lenet.C3, C1_COUT, C3_COUT,
                    C3_error, CONV_SIZE, S2_OUTSIZE, C3_OUTSIZE, delta.C3);
  pool_backward    (C3_error, C1_COUT, S2_error, POOL_STRIDE,
                    C1_OUTSIZE, S2_max_map);
  //test_weight(delta.C1[5][0], CONV_SIZE, CONV_SIZE);
  conv_backward    (S2_error, img_batch[i], lenet.C1, C1_CIN, C1_COUT,
                    C1_error, CONV_SIZE, C1_INSIZE, C1_OUTSIZE, delta.C1);
  //test_weight(delta.C1[5][0], CONV_SIZE, CONV_SIZE);
}


int main(int argc, char** argv){
    init_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte", mnist_train_imgs, mnist_train_labels);
    initialize_lenet();
    int batchindice[BATCHSIZE];
    RandomChoices(batchindice, 60000, BATCHSIZE);
    img_batch = form_img_batch(batchindice, BATCHSIZE, mnist_train_imgs);
    label_batch = form_label_batch(batchindice, BATCHSIZE, mnist_train_labels);
    for(int i=0; i<BATCHSIZE; i++) // change to batchsize eventually
    {
      forward(i);
      backward(i);
    }
    return 0;
}
