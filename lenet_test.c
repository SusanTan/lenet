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
