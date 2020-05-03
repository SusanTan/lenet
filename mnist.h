#include <stdio.h>
#include <stdint.h>
//typedef float** Img;
float** mnist_train_imgs[60000];
uint8_t mnist_train_labels[60000];

void init_data(const char* imagefile, const char* labelfile, float*** imgs, uint8_t* labels, int size);

void form_img_batch(float**** imgs, int start, int batchsize, float*** mnist_train_imgs);
void form_label_batch(uint8_t* labels, int start, int batchsize, uint8_t* mnist_train_labels);
float**** allocate_img_batch(int batchsize);
uint8_t* allocate_label_batch(int batchsize);
void free_image_batch(float**** ptr, int batchsize);
