#include "mnist.h"
#include <stdlib.h>

void maxpool2d_forward(int stride, int pool_size, Img** in, int batchsize, int channels, int img_size, Img** output)
{
  for(int i=0; i<batchsize; i++)
    for(int j=0; j<channels; j++)
      for(int m=0; m<img_size; m++)
        for(int n=0; n<img_size; n++)
          output[i][j][m][n]=0.0f;

  for(int i=0; i<batchsize; i++)
  {
    for(int j=0; j<channels; j++)
    {
      for(int m=0; m<img_size*stride; m+=stride)
      {
        for(int n=0; n<img_size*stride; n+=stride)
        {
          float max = 0.0f;
          for(int l=0; l<pool_size; l++)
            for(int w=0; w<pool_size; w++)
              if(max<in[i][j][m+l][n+w])
                max = in[i][j][m+l][n+w];
          output[i][j][m/stride][n/stride] = max;
        }
      }
    }
  }
}
