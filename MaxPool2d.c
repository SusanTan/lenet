#include "mnist.h"
#include <stdlib.h>

void maxpool2d_forward(int stride, int pool_size, float*** in, int channels, int img_size, float*** output, float*** max_map)
{
  for(int i=0; i<channels; i++)
    for(int j=0; j<img_size; j++)
      for(int k=0; k<img_size; k++)
        output[i][j][k]=0.0f;

  //clear the max mapping for backprop
  for(int i=0; i<channels; i++)
    for(int j=0; j<img_size*stride; j++)
      for(int k=0; k<img_size*stride; k++)
        max_map[i][j][k]=0.0f;

  for(int j=0; j<channels; j++)
  {
    for(int m=0; m<img_size*stride; m+=stride)
    {
      for(int n=0; n<img_size*stride; n+=stride)
      {
        float max = 0.0f;
        int max_i = 0;
        int max_j = 0;
        for(int l=0; l<pool_size; l++)
          for(int w=0; w<pool_size; w++)
            if(max<in[j][m+l][n+w])
            {
              max = in[j][m+l][n+w];
              max_i = m+l;
              max_j = n+w;
            }
        output[j][m/stride][n/stride] = max;
        max_map[j][max_i][max_j] = 1.0f;
       }
     }
   }
}

void pool_backward(float*** error_l_plus_1, int channels, float*** error_l, int stride, int img_size_in, float*** max_map)
{
  //note: += because it's batched
  for(int i=0; i<channels; i++)
    for(int j=0; j<img_size_in; j++)
      for(int k=0; k<img_size_in; k++)
        if(max_map[i][j][k] != 0.0f)
          error_l[i][j][k] = error_l_plus_1[i][j/stride][j/stride];
        else
          error_l[i][j][k] = 0.0f;
}
