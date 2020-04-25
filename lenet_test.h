void test_initialization (LeNet* lenet);
void test_image_batch    (Img**  imgs, int batchsize, int imgsize);
void test_output (Img** imgs, int batchsize);
void test_weight (float** W, int in_channels, int out_channels);
void test_bias (float* B, int out_channels);
