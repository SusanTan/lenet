#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "mnist.h"

#define TRAIN_SIZE 60000
#define IMAGE_SIZE 32

uint8_t raw_imgs[TRAIN_SIZE][28][28];
unsigned char raw_labels[TRAIN_SIZE];

// data is enlarged to 32x32, normalized.
void init_data(const char* imagefile, const char* labelfile, Img* imgs, uint8_t* labels)
{
    //malloc 32x32 for each image;
    for (int k=0; k<TRAIN_SIZE; k++)
    {
      float** img = (float**)malloc(sizeof(float*)*IMAGE_SIZE);
      for(int i=0; i<IMAGE_SIZE; i++)
        img[i] = (float*) malloc (sizeof(float)*IMAGE_SIZE);
      imgs[k] = img;
    }


    //read from input ubyte files
    FILE* imagefp = fopen(imagefile, "rb");
    FILE* labelfp = fopen(labelfile, "rb");
    fseek(imagefp, 16, SEEK_SET);
    fseek(labelfp, 8,  SEEK_SET);
    fread(raw_imgs, sizeof(raw_imgs), 1, imagefp);
    fread(raw_labels, sizeof(raw_labels), 1, labelfp);
    fclose(imagefp);
    fclose(labelfp);

    //enlarge input from 28x28 to 32x32
    for(int k=0; k<TRAIN_SIZE; k++)
    {
      labels[k] = raw_labels[k];
      for(int i=0; i<IMAGE_SIZE; i++)
      {
        for(int j=0; j<IMAGE_SIZE; j++)
        {
          if((i<2 || i>29 || j<2 || j>29)) //assuming it's 32x32
          {
             imgs[k][i][j] = 0;
             continue;
          }
          float normalized =(float) raw_imgs[k][i-2][j-2]/255.0f;
          imgs[k][i][j] = normalized;
        }
      }
    }
}


Img** form_img_batch(int* batchindice, int batchsize, Img* mnist_train_imgs)
{
  //shape: (batch, 1(in_channel), w, h)
  Img** imgs = (Img **)malloc(sizeof(Img*)*batchsize);
  for(int i=0; i<batchsize; i++)
  {
      imgs[i] = (Img*) malloc (sizeof(Img));
      imgs[i][0] = mnist_train_imgs[batchindice[i]];
  }
  return imgs;
}

uint8_t* form_label_batch(int* batchindice, int batchsize, uint8_t* mnist_train_labels)
{
  uint8_t* labels = (uint8_t*)malloc(sizeof(uint8_t)*batchsize);
  for(int i=0; i<batchsize; i++)
      labels[i] = mnist_train_labels[batchindice[i]];
  return labels;
}
