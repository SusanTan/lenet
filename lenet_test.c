#include <stdio.h>
#include "lenet.h"

void test_conv_init(float**** kernel, int in_channels, int out_channels, int size)
{
  for (int i=0; i<out_channels; i++)
    for(int j=0; j<in_channels; j++)
      for(int k=0; k<size; k++)
        for(int l=0; l<size; l++)
            printf("%.3f, ", kernel[i][j][k][l]);
}

void test_linear_init(float** W, float* B, int out, int in)
{
   printf("Weight:\n");
   for(int i=0; i<out; i++)
   {
     for(int j=0; j<in; j++)
     {
       printf("%.3f, ", W[i][j]);
     }
   }
   printf("\nBias:\n");
   for(int i=0; i<out; i++)
     printf("%.3f, ", B[i]);
}

void test_initialization (LeNet* lenet)
{
  printf("Printing C1 Initialized weights..\n");
  test_conv_init(lenet->C1, 1,6,5);
  printf("\nPrinting C3 Initialized weights..\n");
  test_conv_init(lenet->C3, 6,16,5);
  printf("\nPrinting F6 Initialized weights..\n");
  test_linear_init(lenet->F6_W, lenet->F6_B, 84, 120);
}

void test_image_batch (Img** imgs, int batchsize, int imgsize)
{
  printf("\nPrinting the image from channel 0 from 32 batches:\n");
  //just use the first channel
  for(int i=0; i<batchsize; i++)
  {
      printf("\n********Printing batch %d *********\n", i);
      for(int j=0; j<imgsize; j++)
        for(int k=0; k<imgsize; k++)
          printf("%.4f, ",imgs[i][0][j][k]);
  }
}

void test_output (Img** imgs, int batchsize)
{

  for(int i=0; i<batchsize; i++)
  {
    printf("\n*******test batch %d**************\n ", i);
    for(int j=0; j<10; j++)
      printf("%.3f, ", imgs[i][j][0][0]);
  }
  printf("\n");

}


void test_weight (float** W, int in_channels, int out_channels)
{
  printf("\n********test weight*********\n");
  for(int i=0; i<out_channels; i++)
    for(int j=0; j<in_channels; j++)
      printf("%.3f, ", W[i][j]);
}

void test_bias (float* B, int out_channels)
{
  printf("\n********test bias*********\n");
  for(int i=0; i<out_channels; i++)
     printf("%.3f, ", B[i]);
}
