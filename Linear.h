
typedef struct Linear
{
  float** W;
  float*  B;
}Linear;

void initialize_linear(Linear* linear, int in, int out);
