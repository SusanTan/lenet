#include <stdio.h>
#include <stdlib.h>
#include "lenet.h"
#include "lenet_test.h"
#include <math.h>
#include <string.h>

int ntrain = 60000;
int ntest  = 10000;
int traininput_size;//profile purpose
int testinput_size;//profile purpose
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

float*** initialize_images(int d1, int d2, int d3)
{
  float*** images = (float***)malloc(sizeof(float**)*d1);
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

void free_images(float*** img, int d1, int d2)
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
  lenet.C1 = initialize_4Dzeros(C1_COUT, C1_CIN,  CONV_SIZE, CONV_SIZE);
  kaiming_uniform(lenet.C1, C1_CIN,  C1_COUT, CONV_SIZE);
  lenet.C3 = initialize_4Dzeros(C3_COUT, C1_COUT, CONV_SIZE, CONV_SIZE);
  kaiming_uniform(lenet.C3, C1_COUT,  C3_COUT, CONV_SIZE);
  lenet.C5 = initialize_4Dzeros(C5_COUT, C3_COUT, CONV_SIZE, CONV_SIZE);
  kaiming_uniform(lenet.C5, C3_COUT,  C5_COUT, CONV_SIZE);
  lenet.F6_W = initialize_zeroweights(F6_COUT, C5_COUT);
  uniform_W(lenet.F6_W, C5_COUT, F6_COUT);
  lenet.OL_W = initialize_zeroweights(OL_COUT, F6_COUT);
  uniform_W(lenet.OL_W, F6_COUT, OL_COUT);
  lenet.F6_B = initialize_zerobias(F6_COUT);
  uniform_B(lenet.F6_B, C5_COUT, F6_COUT);
  lenet.OL_B = initialize_zerobias(OL_COUT);
  uniform_B(lenet.OL_B, F6_COUT, OL_COUT);

  //initialize all intermediate storages
  out.C1     = initialize_images(C1_COUT, C1_OUTSIZE, C1_OUTSIZE);
  S2_max_map = initialize_images(C1_COUT, C1_OUTSIZE, C1_OUTSIZE);
  out.S2     = initialize_images(C1_COUT, S2_OUTSIZE, S2_OUTSIZE);
  out.C3     = initialize_images(C3_COUT, C3_OUTSIZE, C3_OUTSIZE);
  S4_max_map = initialize_images(C3_COUT, C3_OUTSIZE, C3_OUTSIZE);
  out.S4     = initialize_images(C3_COUT, S4_OUTSIZE, S4_OUTSIZE);
  out.C5     = initialize_images(C5_COUT, C5_OUTSIZE, C5_OUTSIZE);
  out.F6     = initialize_images(F6_COUT, F6_OUTSIZE, F6_OUTSIZE);
  out.OL     = initialize_images(OL_COUT, OL_OUTSIZE, OL_OUTSIZE);
  last_error = initialize_images(OL_COUT, OL_OUTSIZE, OL_OUTSIZE);
  error.OL   = initialize_images(F6_COUT, F6_OUTSIZE, F6_OUTSIZE);
  error.F6   = initialize_images(C5_COUT, C5_OUTSIZE, C5_OUTSIZE);
  error.C5   = initialize_images(C3_COUT, S4_OUTSIZE, S4_OUTSIZE);
  error.S4   = initialize_images(C3_COUT, C3_OUTSIZE, C3_OUTSIZE);
  error.C3   = initialize_images(C1_COUT, S2_OUTSIZE, S2_OUTSIZE);
  error.S2   = initialize_images(C1_COUT, C1_OUTSIZE, C1_OUTSIZE);
  error.C1   = initialize_images(C1_CIN , C1_INSIZE , C1_INSIZE );

  delta.OL_W = initialize_zeroweights(OL_COUT, F6_COUT);
  delta.F6_W = initialize_zeroweights(F6_COUT, C5_COUT);
  delta.OL_B = initialize_zerobias(OL_COUT);
  delta.F6_B = initialize_zerobias(F6_COUT);
  delta.C5   = initialize_4Dzeros(C5_COUT, C3_COUT, CONV_SIZE, CONV_SIZE);
  delta.C3   = initialize_4Dzeros(C3_COUT, C1_COUT, CONV_SIZE, CONV_SIZE);
  delta.C1   = initialize_4Dzeros(C1_COUT, C1_CIN , CONV_SIZE, CONV_SIZE);
}

void torch_tanh (float*** x, int img_size, int channels)
{
  for(int i=0; i<channels; i++)
    for(int j=0; j<img_size; j++)
      for(int k=0; k<img_size; k++)
        x[i][j][k] = tanh(x[i][j][k]);
}

void forward(float*** image)
{
  conv2d_forward(lenet.C1, image, C1_OUTSIZE, C1_CIN, C1_COUT, out.C1);
  torch_tanh(out.C1, C1_OUTSIZE, C1_COUT);
	maxpool2d_forward(POOL_STRIDE, POOL_SIZE, out.C1, C1_COUT, S2_OUTSIZE, out.S2, S2_max_map);
  conv2d_forward(lenet.C3, out.S2, C3_OUTSIZE, C1_COUT, C3_COUT, out.C3);
  torch_tanh(out.C3, C3_OUTSIZE, C3_COUT);
  maxpool2d_forward(POOL_STRIDE, POOL_SIZE, out.C3, C3_COUT, S4_OUTSIZE, out.S4, S4_max_map);
  conv2d_forward(lenet.C5, out.S4, C5_OUTSIZE, C3_COUT, C5_COUT, out.C5);
  torch_tanh(out.C5, C5_OUTSIZE, C5_COUT);
  linear_forward(lenet.F6_W, lenet.F6_B, out.C5, C5_COUT, F6_COUT, out.F6);
  torch_tanh(out.F6, F6_OUTSIZE, F6_COUT);
  linear_forward(lenet.OL_W, lenet.OL_B, out.F6, F6_COUT, OL_COUT, out.OL);
  torch_tanh(out.OL, OL_OUTSIZE, OL_COUT);
}

void backward(i)
{
  last_layer_prep  (train_label_batch[i], out.OL, OL_COUT, last_error);
  linear_backward  (last_error, out.F6, lenet.OL_W, lenet.OL_B,
                    F6_COUT, OL_COUT, error.OL, delta.OL_W, delta.OL_B);
  linear_backward  (error.OL, out.C5, lenet.F6_W, lenet.F6_B,
                    C5_COUT, F6_COUT, error.F6, delta.F6_W, delta.F6_B);
  conv_backward    (error.F6, out.S4, lenet.C5, C3_COUT, C5_COUT,
                    error.C5, CONV_SIZE, S4_OUTSIZE, C5_OUTSIZE, delta.C5);
  pool_backward    (error.C5, C3_COUT, error.S4, POOL_STRIDE,
                    C3_OUTSIZE, S4_max_map);
  conv_backward    (error.S4, out.S2, lenet.C3, C1_COUT, C3_COUT,
                    error.C3, CONV_SIZE, S2_OUTSIZE, C3_OUTSIZE, delta.C3);
  pool_backward    (error.C3, C1_COUT, error.S2, POOL_STRIDE,
                    C1_OUTSIZE, S2_max_map);
  conv_backward    (error.S2, train_img_batch[i], lenet.C1, C1_CIN, C1_COUT,
                    error.C1, CONV_SIZE, C1_INSIZE, C1_OUTSIZE, delta.C1);
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
  for(int j=0; j<ntrain/BATCHSIZE; j++)
  {
    RandomChoices(batchindice, traininput_size, BATCHSIZE);
    form_img_batch(train_img_batch,batchindice, BATCHSIZE, mnist_train_imgs);
    form_label_batch(train_label_batch, batchindice, BATCHSIZE, mnist_train_labels);

    for(int i=0; i<BATCHSIZE; i++) // change to batchsize eventually
    {
      forward(train_img_batch[i]);
      backward(i);
    }
    weight_update();
  }
  free_image_batch(train_img_batch, BATCHSIZE);
  free(train_label_batch);
}


void testing()
{
  int batchindice[1];
  test_img_batch   = allocate_img_batch(1);
  test_label_batch = allocate_label_batch(1);
  int errors = 0;
  for(int j=0; j<ntest; j++)
  {
    RandomChoices(batchindice, testinput_size, 1);
    form_img_batch(test_img_batch, batchindice, 1, mnist_test_imgs);
    form_label_batch(test_label_batch, batchindice, 1, mnist_test_labels);
    forward(test_img_batch[0]);
    int pred_digit = 0;
    int actual_digit = 0;
    float max = -100.0f;
    for(int i=0; i<10; i++)
    {
      if(out.OL[i][0][0]>max)
      {
        max = out.OL[i][0][0];
        pred_digit = i;
      }
      if(test_label_batch[0]==i)
        actual_digit = i;
    }
    if(pred_digit != actual_digit)
      errors++;
  }
  free_image_batch(test_img_batch, 1);
  free(test_label_batch);
  double err =(double) errors/(double)ntest;
  printf("\n classification error: %.3f\n", err);
}


void free_all()
{
  free_conv(lenet.C1, C1_CIN,  C1_CIN,  CONV_SIZE);
  free_conv(lenet.C3, C1_COUT, C3_COUT, CONV_SIZE);
  free_conv(lenet.C5, C3_COUT, C5_COUT, CONV_SIZE);
  free_linear(lenet.F6_W, lenet.F6_B, F6_COUT);
  free_linear(lenet.OL_W, lenet.OL_B, OL_COUT);
  free_images(out.C1, C1_COUT, C1_OUTSIZE);
  free_images(S2_max_map, C1_COUT, C1_OUTSIZE);
  free_images(out.S2, C1_COUT, S2_OUTSIZE);
  free_images(out.C3, C3_COUT, C3_OUTSIZE);
  free_images(S4_max_map, C3_COUT, C3_OUTSIZE);
  free_images(out.S4, C3_COUT, S4_OUTSIZE);
  free_images(out.C5, C5_COUT, C5_OUTSIZE);
  free_images(out.F6, F6_COUT, F6_OUTSIZE);
  free_images(out.OL, OL_COUT, OL_OUTSIZE);
  free_images(last_error, OL_COUT, OL_OUTSIZE);
  free_images(error.OL, F6_COUT, F6_OUTSIZE);
  free_images(error.F6, C5_COUT, C5_OUTSIZE);
  free_images(error.C5, C3_COUT, S4_OUTSIZE);
  free_images(error.S4, C3_COUT, C3_OUTSIZE);
  free_images(error.C3, C1_COUT, S2_OUTSIZE);
  free_images(error.S2, C1_COUT, C1_OUTSIZE);
  free_images(error.C1, C1_CIN , C1_INSIZE);
  free_conv(delta.C5, C3_COUT, C5_COUT, CONV_SIZE);
  free_conv(delta.C3, C1_COUT, C3_COUT, CONV_SIZE);
  free_conv(delta.C1, C1_CIN,  C1_COUT, CONV_SIZE);
  free_linear(delta.F6_W, delta.F6_B, F6_COUT);
  free_linear(delta.OL_W, delta.OL_B, OL_COUT);
}

int main(int argc, char** argv){
    if(argc < 3)
    {
      printf("please enter profile/run, then train size, then test size\n");
      exit(-1);
    }
    ntrain = atoi(argv[2]);
    ntest  = atoi(argv[3]);
    if(strcmp(argv[1], "profile") == 0)
    {
      init_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", mnist_train_imgs, mnist_train_labels, 1000);
      init_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", mnist_test_imgs, mnist_test_labels, 1000);
      traininput_size = 1000;
      testinput_size = 1000;
    }
    else if(strcmp(argv[1], "run") == 0)
    {
      init_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte", mnist_train_imgs, mnist_train_labels, 60000);
      init_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", mnist_test_imgs, mnist_test_labels, 10000);
      traininput_size = 60000;
      testinput_size = 10000;
    }
    else
      printf("fisrt argument should be keyword profile or run\n");
    initialize_lenet();
    training();
    testing();
    free_all();
    return 0;
}
