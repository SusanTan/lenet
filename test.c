#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "mnist.h"

#define TRAIN_SIZE 60000
#define IMAGE_SIZE 32

uint8_t raw_imgs[TRAIN_SIZE][28][28];
unsigned char raw_labels[TRAIN_SIZE];

int main(int argc, char** argv)
{


    FILE* imagefp = fopen("train-images-idx3-ubyte", "rb");
    FILE* labelfp = fopen("train-labels-idx1-ubyte", "rb");
    fseek(imagefp, 16, SEEK_SET);
    fseek(labelfp, 8,  SEEK_SET);
    fread(raw_imgs, sizeof(raw_imgs), 1, imagefp);
    fread(raw_labels, sizeof(raw_labels), 1, labelfp);

    for(int i=0; i< 28; i++)
      for(int j=0; j<28; j++)
        printf("%d, ", raw_imgs[59999][i][j]);

    return 0;

}
