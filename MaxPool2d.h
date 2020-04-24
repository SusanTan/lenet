
typedef struct MaxPool2d
{
  int stride;
  int width;
  int length;
} MaxPool2d;

void initialize_pool(MaxPool2d* pool, int width, int length, int stride);


