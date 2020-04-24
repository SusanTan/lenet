#include <stdio.h>
#include <stdint.h>
typedef float** Img;
Img mnist_train_imgs[60000];
uint8_t mnist_train_labels[60000];

void init_data(const char* imagefile, const char* labelfile, Img* imgs, uint8_t* labels);

Img** form_img_batch(int* batchindice, int batchsize, Img* mnist_train_imgs);
uint8_t* form_label_batch(int* batchindice, int batchsize, uint8_t* mnist_train_labels);
