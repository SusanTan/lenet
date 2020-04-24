#include <stdio.h>
#include "lenet.h"

#ifndef CONV_W
#define CONV_W 5
#endif
#ifndef CONV_L
#define CONV_L 5
#endif
void test_conv_init(Conv2d conv)
{
  for (int i=0; i<conv.out_channels; i++)
  {
    for(int j=0; j<conv.in_channels; j++)
    {
      for(int k=0; k<CONV_W; k++)
      {
        for(int l=0; l<CONV_L; l++)
        {
            printf("%.3f, ", conv.kernel[i][j][k][l]);
        }
      }
    }
  }
}

void test_linear_init(Linear l, int out, int in)
{
   printf("Weight:\n");
   for(int i=0; i<out; i++)
   {
     for(int j=0; j<in; j++)
     {
       printf("%.3f, ", l.W[i][j]);
     }
   }
   printf("\nBias:\n");
   for(int i=0; i<out; i++)
     printf("%.3f, ", l.B[i]);
}

void test_initialization (LeNet* lenet)
{
  printf("Printing C1 Initialized weights..\n");
  test_conv_init(lenet->C1);
  printf("\nPrinting C3 Initialized weights..\n");
  test_conv_init(lenet->C3);
  printf("\nPrinting F6 Initialized weights..\n");
  test_linear_init(lenet->F6, 84, 120);
}
