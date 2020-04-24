#include "mnist.h"
#include <stdlib.h>

Img** maxpool2d_forward(int stride, int pool_size, Img** in, int batchsize, int channels, int img_size)
{
  int size_out = (img_size-pool_size)/stride + 1;
  Img** forward = (Img**)malloc(sizeof(Img*)*batchsize);
  for(int i=0; i<batchsize; i++)
    forward[i] = (Img*)malloc(sizeof(Img)*channels);

  for(int i=0; i<batchsize; i++)
  {
    for(int j=0; j<channels; j++)
    {

      float** result = (float**)malloc(sizeof(float*)*size_out);
      for(int k=0; k<size_out; k++)
         result[k] = (float*)malloc(sizeof(float)*size_out);

      for(int k=0; k<size_out; k++)
        for(int l=0; l<size_out; l++)
          result[k][l] = 0.0f;

      for(int m=0; m<(img_size-pool_size+1); m+=stride)
      {
        for(int n=0; n<(img_size-pool_size+1); n+=stride)
        {
          int max = 0;
          for(int l=0; l<pool_size; l++)
            for(int w=0; w<pool_size; w++)
              if(max<in[i][j][m+l][n+w])
                max = in[i][j][m+l][n+w];
          result[m/stride][n/stride] = max;
        }
      }

      forward[i][j] = result;
    }
  }

  return forward;
}
