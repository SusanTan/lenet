typedef float** Kernel;
typedef float*  Conv_Bias;

typedef struct Conv2d
{
  int in_channels;
  int out_channels;
  int width;
  int height;
  Kernel** kernel;
} Conv2d;

void initialize_conv(Conv2d* conv, int in_channels, int out_channels, int kernel_width, int kernel_height);
