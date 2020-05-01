#include <stdio.h>
#include <stdlib.h>
#include "lenet.h"
#include "lenet_test.h"
#include <math.h>

void RandomChoices(int* batchindice, int range, int size)
{
  for(int i=0; i<size; i++)
    batchindice[i] = (rand()%(range));
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

void free_images(Img* img, int d1, int d2)
{
  for(int i=0; i<d1; i++)
  {
    for(int j=0; j<d2; j++)
      free(img[i][j]);
    free(img[i]);
  }
  free(img);
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

void forward(Img* image)
{
  conv2d_forward(lenet.C1, image, C1_OUTSIZE, C1_CIN, C1_COUT, C1_out);
  torch_tanh(C1_out, C1_OUTSIZE, C1_COUT);
	maxpool2d_forward(POOL_STRIDE, POOL_SIZE, C1_out, C1_COUT, S2_OUTSIZE, S2_out, S2_max_map);
  conv2d_forward(lenet.C3, S2_out, C3_OUTSIZE, C1_COUT, C3_COUT, C3_out);
  torch_tanh(C3_out, C3_OUTSIZE, C3_COUT);
  maxpool2d_forward(POOL_STRIDE, POOL_SIZE, C3_out, C3_COUT, S4_OUTSIZE, S4_out, S4_max_map);
  conv2d_forward(lenet.C5, S4_out, C5_OUTSIZE, C3_COUT, C5_COUT, C5_out);
  torch_tanh(C5_out, C5_OUTSIZE, C5_COUT);
  linear_forward(lenet.F6_W, lenet.F6_B, C5_out, C5_COUT, F6_COUT, F6_out);
  torch_tanh(F6_out, F6_OUTSIZE, F6_COUT);
  linear_forward(lenet.OL_W, lenet.OL_B, F6_out, F6_COUT, OL_COUT, OL_out);
  torch_tanh(OL_out, OL_OUTSIZE, OL_COUT);
}

void backward(i)
{
  //float loss = mse_loss(output, label_batch);
  last_layer_prep  (train_label_batch[i], OL_out, OL_COUT, last_error);
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
  conv_backward    (S2_error, train_img_batch[i], lenet.C1, C1_CIN, C1_COUT,
                    C1_error, CONV_SIZE, C1_INSIZE, C1_OUTSIZE, delta.C1);
}

void update_conv(float**** W, float**** W_d, int in_c, int out_c, int kernel)
{
  for(int i=0; i<out_c; i++)
    for(int j=0; j<in_c; j++)
      for(int k=0; k<kernel; k++)
        for(int l=0; l<kernel; l++)
        {
          W[i][j][k][l] += W_d[i][j][k][l]*0.2/BATCHSIZE;
          W_d[i][j][k][l] = 0.0f;
        }
}

void update_linear_w(float** W, float** W_d, int in_c, int out_c)
{
  for(int i=0; i<out_c; i++)
    for(int j=0; j<in_c; j++)
    {
      W[i][j] += W_d[i][j]*0.2/BATCHSIZE;
      W_d[i][j] = 0.0f;
    }
}

void update_linear_b(float* B, float* B_d, int out_c)
{
  for(int i=0; i<out_c; i++)
  {
    B[i] += B_d[i]*0.2/BATCHSIZE;
    B_d[i] = 0.0f;
  }
}

void weight_update()
{
  //update, and also clear the buffer
  update_conv(lenet.C1, delta.C1, C1_CIN,  C1_COUT, CONV_SIZE);
  update_conv(lenet.C3, delta.C3, C1_COUT, C3_COUT, CONV_SIZE);
  update_conv(lenet.C5, delta.C5, C3_COUT, C5_COUT, CONV_SIZE);
  update_linear_w(lenet.F6_W, delta.F6_W, C5_COUT, F6_COUT);
  update_linear_w(lenet.OL_W, delta.OL_W, F6_COUT, OL_COUT);
  update_linear_b(lenet.F6_B, delta.F6_B, F6_COUT);
  update_linear_b(lenet.OL_B, delta.OL_B, OL_COUT);
}

void training()
{
  int batchindice[BATCHSIZE];
  train_img_batch   = allocate_img_batch(BATCHSIZE);
  train_label_batch = allocate_label_batch(BATCHSIZE);
  for(int j=0; j<60000/BATCHSIZE; j++)
  {
    RandomChoices(batchindice, 60000, BATCHSIZE);
    form_img_batch(train_img_batch,batchindice, BATCHSIZE, mnist_train_imgs);
    form_label_batch(train_label_batch, batchindice, BATCHSIZE, mnist_train_labels);

    for(int i=0; i<BATCHSIZE; i++) // change to batchsize eventually
    {
      forward(train_img_batch[i]);
      backward(i);
    }
    weight_update();
  }
}


void testing()
{
  int batchindice[1];
  test_img_batch   = allocate_img_batch(BATCHSIZE);
  test_label_batch = allocate_label_batch(BATCHSIZE);
  int errors = 0;
  for(int j=0; j<10000; j++)
  {
    RandomChoices(batchindice, 10000, 1);
    form_img_batch(test_img_batch, batchindice, 1, mnist_test_imgs);
    form_label_batch(test_label_batch, batchindice, 1, mnist_test_labels);
    forward(test_img_batch[0]);
    int pred_digit = 0;
    int actual_digit = 0;
    float max = -100.0f;
    //printf("OL_out values: \n");
    for(int i=0; i<10; i++)
    {
      //printf("%.3f, ", OL_out[i][0][0]);
      if(OL_out[i][0][0]>max)
      {
        max = OL_out[i][0][0];
        pred_digit = i;
      }
      if(test_label_batch[0]==i)
        actual_digit = i;
    }
    if(pred_digit != actual_digit)
      errors++;
    //printf("predicted: %d\n", pred_digit);
    //printf("testing digit: %d\n", actual_digit);
  }
  double err =(double) errors/10000.0;
  printf("\n classification error: %.3f\n", err);
}


void free_all()
{
  free_conv(lenet.C1, C1_CIN,  C1_CIN,  CONV_SIZE);
  free_conv(lenet.C3, C1_COUT, C3_COUT, CONV_SIZE);
  free_conv(lenet.C5, C3_COUT, C5_COUT, CONV_SIZE);
  free_linear(lenet.F6_W, lenet.F6_B, F6_COUT);
  free_linear(lenet.OL_W, lenet.OL_B, OL_COUT);
  free_images(C1_out, C1_COUT, C1_OUTSIZE);
  free_images(S2_max_map, C1_COUT, C1_OUTSIZE);
  free_images(S2_out, C1_COUT, S2_OUTSIZE);
  free_images(C3_out, C3_COUT, C3_OUTSIZE);
  free_images(S4_max_map, C3_COUT, C3_OUTSIZE);
  free_images(S4_out, C3_COUT, S4_OUTSIZE);
  free_images(C5_out, C5_COUT, C5_OUTSIZE);
  free_images(F6_out, F6_COUT, F6_OUTSIZE);
  free_images(OL_out, OL_COUT, OL_OUTSIZE);
  free_images(last_error, OL_COUT, OL_OUTSIZE);
  free_images(OL_error, F6_COUT, F6_OUTSIZE);
  free_images(F6_error, C5_COUT, C5_OUTSIZE);
  free_images(C5_error, C3_COUT, S4_OUTSIZE);
  free_images(S4_error, C3_COUT, C3_OUTSIZE);
  free_images(C3_error, C1_COUT, S2_OUTSIZE);
  free_images(S2_error, C1_COUT, C1_OUTSIZE);
  free_images(C1_error, C1_CIN , C1_INSIZE);
  free_conv(delta.C5, C3_COUT, C5_COUT, CONV_SIZE);
  free_conv(delta.C3, C1_COUT, C3_COUT, CONV_SIZE);
  free_conv(delta.C1, C1_CIN,  C1_COUT, CONV_SIZE);
  free_linear(delta.F6_W, delta.F6_B, F6_COUT);
  free_linear(delta.OL_W, delta.OL_B, OL_COUT);


}

int main(int argc, char** argv){
    init_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte", mnist_train_imgs, mnist_train_labels, 60000);
    init_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", mnist_test_imgs, mnist_test_labels, 10000);
    initialize_lenet();
    //training();
    //testing();
    free_all();
    //TODO: free all the mallocs
    return 0;
}
