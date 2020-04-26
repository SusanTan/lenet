#include "mnist.h"
#include <stdlib.h>

void maxpool2d_forward(int stride, int pool_size, Img** in, int batchsize, int channels, int img_size, Img** output, Img** max_map)
{
  for(int i=0; i<batchsize; i++)
    for(int j=0; j<channels; j++)
      for(int m=0; m<img_size; m++)
        for(int n=0; n<img_size; n++)
          output[i][j][m][n]=0.0f;

  //clear the max mapping for backprop
  for(int i=0; i<batchsize; i++)
    for(int j=0; j<channels; j++)
      for(int m=0; m<img_size*stride; m++)
        for(int n=0; n<img_size*stride; n++)
          max_map[i][j][m][n]=0.0f;

  for(int i=0; i<batchsize; i++)
  {
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
              if(max<in[i][j][m+l][n+w])
              {
                max = in[i][j][m+l][n+w];
                max_i = m+l;
                max_j = n+w;
              }
          output[i][j][m/stride][n/stride] = max;
          max_map[i][j][max_i][max_j] = 1.0f;
        }
      }
    }
  }
}

void pool_backward(Img** delta_l_plus_1, Img** in, int batchsize, int channels, Img** delta_l, int stride, int img_size_in, Img** max_map)
{
//  for(int i=0; i<batchsize; i++)
//    for(int j=0; j<img_size_in; j++)
//      for(int k=0; k<img_size_in; k++)


}
