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

float** initialize_deltas(int d1, int d2)
{
  float** result = (float**)malloc(d1*sizeof(float*));
  for(int i=0; i<d1; i++)
    result[i] = (float*)malloc(d2*sizeof(float));
  return result;
}

Img** initialize_images(int d1, int d2, int d3, int d4)
{
  Img** images = (Img**)malloc(sizeof(Img*)*d1);
  for(int i=0; i<d1; i++)
  {
    images[i] = (Img*)malloc(sizeof(Img)*d2);
    for(int j=0; j<d2; j++)
    {
      images[i][j] = (float**)malloc(sizeof(float*)*d3);
      for(int k=0; k<d3; k++)
        images[i][j][k] = (float*)malloc(sizeof(float)*d4);
    }
  }
  return images;
}

void initialize_lenet(LeNet* lenet){
  lenet->pool_stride = POOL_STRIDE;
  lenet->pool_size   = POOL_SIZE;
  lenet->C1 = initialize_conv(C1_CIN,  C1_COUT, CONV_SIZE);
  lenet->C3 = initialize_conv(C1_COUT, C3_COUT, CONV_SIZE);
  lenet->C5 = initialize_conv(C3_COUT, C5_COUT, CONV_SIZE);
  lenet->F6_W = initialize_linear_weight(C5_COUT, F6_COUT);
  lenet->F6_B = initialize_linear_bias  (C5_COUT, F6_COUT);
  lenet->OL_W = initialize_linear_weight(F6_COUT, OL_COUT);
  lenet->OL_B = initialize_linear_bias  (F6_COUT, OL_COUT);
  //test_initialization (lenet);

  //initialize all intermediate storages
  C1_out     = initialize_images(BATCHSIZE, C1_COUT, C1_OUTSIZE, C1_OUTSIZE);
  S2_out     = initialize_images(BATCHSIZE, C1_COUT, S2_OUTSIZE, S2_OUTSIZE);
  C3_out     = initialize_images(BATCHSIZE, C3_COUT, C3_OUTSIZE, C3_OUTSIZE);
  S4_out     = initialize_images(BATCHSIZE, C3_COUT, S4_OUTSIZE, S4_OUTSIZE);
  C5_out     = initialize_images(BATCHSIZE, C5_COUT, C5_OUTSIZE, C5_OUTSIZE);
  F6_out     = initialize_images(BATCHSIZE, F6_COUT, F6_OUTSIZE, F6_OUTSIZE);
  OL_out     = initialize_images(BATCHSIZE, OL_COUT, OL_OUTSIZE, OL_OUTSIZE);
  last_delta = initialize_images(BATCHSIZE, OL_COUT, OL_OUTSIZE, OL_OUTSIZE);
  OL_delta   = initialize_images(BATCHSIZE, F6_COUT, F6_OUTSIZE, F6_OUTSIZE);
  F6_delta   = initialize_images(BATCHSIZE, C5_COUT, C5_OUTSIZE, C5_OUTSIZE);
  C5_delta   = initialize_images(BATCHSIZE, C3_COUT, S4_OUTSIZE, S4_OUTSIZE);
}

void torch_tanh (Img** x, int batchsize, int img_size, int channels)
{
  for(int i=0; i<batchsize; i++)
    for(int j=0; j<channels; j++)
      for(int k=0; k<img_size; k++)
        for(int l=0; l<img_size; l++)
          x[i][j][k][l] = tanh(x[i][j][k][l]);
}

void forward()
{
  conv2d_forward(lenet.C1, img_batch, BATCHSIZE, C1_OUTSIZE, C1_CIN, C1_COUT, C1_out);
  torch_tanh(C1_out, BATCHSIZE, C1_OUTSIZE, C1_COUT);
	maxpool2d_forward(lenet.pool_stride, lenet.pool_size, C1_out, BATCHSIZE, C1_COUT, S2_OUTSIZE, S2_out);
  conv2d_forward(lenet.C3, S2_out, BATCHSIZE, C3_OUTSIZE, C1_COUT, C3_COUT, C3_out);
  torch_tanh(C3_out, BATCHSIZE, C3_OUTSIZE, C3_COUT);
  maxpool2d_forward(lenet.pool_stride, lenet.pool_size, C3_out, BATCHSIZE, C3_COUT, S4_OUTSIZE, S4_out);
  conv2d_forward(lenet.C5, S4_out, BATCHSIZE, C5_OUTSIZE, C3_COUT, C5_COUT, C5_out);
  torch_tanh(C5_out, BATCHSIZE, C5_OUTSIZE, C5_COUT);
  linear_forward(lenet.F6_W, lenet.F6_B, C5_out, BATCHSIZE, C5_COUT, F6_COUT, F6_out);
  torch_tanh(F6_out, BATCHSIZE, F6_OUTSIZE, F6_COUT);
  linear_forward(lenet.OL_W, lenet.OL_B, F6_out, BATCHSIZE, F6_COUT, OL_COUT, OL_out);
  torch_tanh(OL_out, BATCHSIZE, OL_OUTSIZE, OL_COUT);
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

void backward()
{
  //float loss = mse_loss(output, label_batch);
  last_layer_prep  (label_batch, OL_out, BATCHSIZE, OL_COUT, last_delta);
  linear_backward  (last_delta, F6_out, lenet.OL_W, lenet.OL_B, BATCHSIZE,
                    F6_COUT, OL_COUT, OL_delta);
  linear_backward  (OL_delta, C5_out, lenet.F6_W, lenet.F6_B, BATCHSIZE,
                    C5_COUT, F6_COUT, F6_delta);
  test_weight(lenet.C5[119][15], CONV_SIZE, CONV_SIZE);
  conv_backward    (F6_delta, S4_out, lenet.C5, BATCHSIZE, C3_COUT, C5_COUT,
                    C5_delta, CONV_SIZE, S4_OUTSIZE, C5_OUTSIZE);
  test_weight(lenet.C5[119][15], CONV_SIZE, CONV_SIZE);
}


int main(int argc, char** argv){
    init_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte", mnist_train_imgs, mnist_train_labels);
    initialize_lenet(&lenet);

    //form input images with size(batch, Cin, h, w);
    int batchindice[BATCHSIZE];
    RandomChoices(batchindice, 60000, BATCHSIZE);
    img_batch = form_img_batch(batchindice, BATCHSIZE, mnist_train_imgs);
    label_batch = form_label_batch(batchindice, BATCHSIZE, mnist_train_labels);
    forward();
    test_output(OL_out, BATCHSIZE);
    backward();
    return 0;
}
