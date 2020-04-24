#include <stdio.h>
#include <stdlib.h>
#include "lenet.h"
#include "lenet_test.h"
#include <math.h>

#define CONV_W      5
#define CONV_L      5
#define POOL_W      2
#define POOL_L      2
#define POOL_STRIDE 2
#define BATCHSIZE   32

//Channel size
#define C1_CIN      1
#define C1_COUT     6
#define C3_CIN      6
#define C3_COUT     16
#define C5_COUT     120
#define F6_COUT     84
#define OL_COUT     10

//image size
#define C1_INSIZE   32
#define S2_INSIZE   28
#define C3_INSIZE   14
#define S4_INSIZE   10
#define C5_INSIZE   5
#define F6_INSIZE   1


void RandomChoices(int* batchindice, int range, int size)
{
  for(int i=0; i<size; i++)
    batchindice[i] = (rand()%(range+1));
}


void initialize_lenet(LeNet* lenet){
  initialize_conv(&lenet->C1, C1_CIN, C1_COUT, CONV_W, CONV_L);
  initialize_pool(&lenet->S2, POOL_W, POOL_L, 2);
  initialize_conv(&lenet->C3, C3_CIN, C3_COUT, CONV_W, CONV_L);
  initialize_pool(&lenet->S4, POOL_W, POOL_L, 2);
  initialize_conv(&lenet->C5, 16, 120, CONV_W, CONV_L);
  initialize_linear(&lenet->F6, 120, 84);
  initialize_linear(&lenet->OL, 84 , 10);
  test_initialization (lenet);
}

void torch_tanh (Imgs x, int batchsize, int img_size, int channels)
{
  for(int i=0; i<batchsize; i++)
    for(int j=0; j<channels; j++)
      for(int k=0; k<img_size; k++)
        for(int l=0; l<img_size; l++)
          x[i][j].image[k][l] = tanh(x[i][j].image[k][l]);
}

void forward(Imgs x)
{
  Imgs C1_out = conv2d_forward(&lenet.C1, x, BATCHSIZE, C1_INSIZE);
  torch_tanh(C1_out, BATCHSIZE, S2_INSIZE, C1_COUT);
	Imgs S2_out = maxpool2d_forward(lenet.S2, C1_out, BATCHSIZE, C1_COUT, S2_INSIZE);
  Imgs C3_out = conv2d_forward(&lenet.C3, S2_out, BATCHSIZE, C3_INSIZE);
  torch_tanh(C3_out, BATCHSIZE, S4_INSIZE, C3_COUT);
  Imgs S4_out = maxpool2d_forward(lenet.S4, C3_out, BATCHSIZE, C3_COUT, S4_INSIZE);
  Imgs C5_out = conv2d_forward(&lenet.C5, S4_out, BATCHSIZE, C5_INSIZE);
  torch_tanh(C5_out, BATCHSIZE, F6_INSIZE, C5_COUT);
  Imgs F6_out = linear_forward(lenet.F6, C5_out, BATCHSIZE, C5_COUT, F6_COUT);
  Imgs output = linear_forward(lenet.OL, F6_out, BATCHSIZE, F6_COUT, OL_COUT);
}

int main(int argc, char** argv){

    if (argc != 2){
       printf("Wrong input format! Aborted.\n");
       exit(1);
    }

    FILE *fp;
    fp = fopen(argv[1],"r");

    if (fp == NULL){
      printf("Require a csv file. Input is not a file. Aborted.");
      exit(2);
    }

    init_data(fp, mnist, 32, 32);
    initialize_lenet(&lenet);

    //form input images with size(batch, Cin, h, w);
    int batchindice[BATCHSIZE];
    RandomChoices(batchindice, 60000, BATCHSIZE);
    Imgs x = form_images(batchindice, BATCHSIZE, 1);
    forward(x);
    return 0;
}
