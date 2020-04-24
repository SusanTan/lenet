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
  Img** C1_out = conv2d_forward(lenet.C1, img_batch, BATCHSIZE, C1_INSIZE, C1_CIN, C1_COUT);
  //test_image_batch(C1_out, BATCHSIZE, S2_INSIZE);
  torch_tanh(C1_out, BATCHSIZE, S2_INSIZE, C1_COUT);
  //test_image_batch(C1_out, BATCHSIZE, S2_INSIZE);
	Img** S2_out = maxpool2d_forward(lenet.pool_stride, lenet.pool_size, C1_out, BATCHSIZE, C1_COUT, S2_INSIZE);
  test_image_batch(S2_out, BATCHSIZE, C3_INSIZE);
  Img** C3_out = conv2d_forward(lenet.C3, S2_out, BATCHSIZE, C3_INSIZE, C1_COUT, C3_COUT);
  torch_tanh(C3_out, BATCHSIZE, S4_INSIZE, C3_COUT);
  Img** S4_out = maxpool2d_forward(lenet.pool_stride, lenet.pool_size, C3_out, BATCHSIZE, C3_COUT, S4_INSIZE);
  Img** C5_out = conv2d_forward(lenet.C5, S4_out, BATCHSIZE, C5_INSIZE, C3_COUT, C5_COUT);
  torch_tanh(C5_out, BATCHSIZE, F6_INSIZE, C5_COUT);
  Img** F6_out = linear_forward(lenet.F6_W, lenet.F6_B, C5_out, BATCHSIZE, C5_COUT, F6_COUT);
  Img** output = linear_forward(lenet.OL_W, lenet.OL_B, F6_out, BATCHSIZE, F6_COUT, OL_COUT);
  for(int i=0; i<32; i++)
  {
    printf("test batch %d:\n ", i);
    for(int j=0; j<10; j++)
      printf("%.3f, ", output[i][j][0][0]);
  }
}

int main(int argc, char** argv){
    init_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte", mnist_train_imgs, mnist_train_labels);
    initialize_lenet(&lenet);

    //form input images with size(batch, Cin, h, w);
    int batchindice[BATCHSIZE];
    RandomChoices(batchindice, 60000, BATCHSIZE);
    img_batch = form_img_batch(batchindice, BATCHSIZE, mnist_train_imgs);
    uint8_t* label_batch = form_label_batch(batchindice, BATCHSIZE, mnist_train_labels);
    //test_image_batch(img_batch, BATCHSIZE, 32);
    forward();
    return 0;
}
