#include <stdio.h>
#include <stdint.h>
typedef float** Img;
Img mnist_train_imgs[60000];
uint8_t mnist_train_labels[60000];

void init_data(const char* imagefile, const char* labelfile, Img* imgs, uint8_t* labels, int size);

void form_img_batch(Img** imgs, int* batchindice, int batchsize, Img* mnist_train_imgs);
void form_label_batch(uint8_t* labels, int* batchindice, int batchsize, uint8_t* mnist_train_labels);
Img** allocate_img_batch(int batchsize);
uint8_t* allocate_label_batch(int batchsize);
void free_image_batch(Img** ptr, int batchsize);
