#include <stdio.h>

typedef struct mnist_data {
  float** image; //maximum size
  unsigned int  label;
}mnist_data;

typedef mnist_data** Imgs;

mnist_data mnist[60000];

void init_data(FILE* fp, mnist_data* mnist, int h, int w);

Imgs form_images(int* batchindice, int batchsize, int channels);
