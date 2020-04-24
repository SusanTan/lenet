#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mnist.h"


// data is enlarged to 32x32, normalized.
void init_data(FILE* fp, mnist_data* mnist, int h, int w)
{
    //malloc 32x32 for each image;
    for (int k=0; k<60000; k++)
    {
      float** img = (float**) malloc (sizeof(float*)*h);
      for(int i=0; i<h; i++)
      {
        img[i] = (float*) malloc (sizeof(float)*w);
      }
      mnist[k].image = img;
    }


    for(int k=0; k<60000; k++)
    {
      char line[4*785];
      fgets(line, 4*785, fp);
      const char* tok;
      mnist[k].label = atoi(strtok(line,","));
      for(int i=0; i<h; i++)
      {
        for(int j=0; j<w; j++)
        {
          if((i<2 || i>29 || j<2 || j>29)) //assuming it's 32x32
          {
             mnist[k].image[i][j] = 0;
            continue;
          }
          tok = strtok(NULL, ",\n");
          float normalized =((float) atoi(tok))/255.0f;
          mnist[k].image[i][j] = normalized;
        }
      }
    }
    fclose(fp);
}


Imgs form_images(int* batchindice, int batchsize, int channels)
{
  Imgs imgs = (mnist_data**)malloc(sizeof(mnist_data*)*batchsize);
  for(int i=0; i<batchsize; i++)
    imgs[i] = (mnist_data*)malloc(sizeof(mnist_data)*channels);

  for(int i=0; i<batchsize; i++)
  {
    for(int j=0; j<channels; j++)
    {
      imgs[i][j] = mnist[batchindice[i]];
    }
  }
  return imgs;
}
