#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "grad.h"


int main(int argc, char **argv) {
  float x = 1.0;
  float y = 2.0;
  float z = x * y;
  printf("z = %f\n", z);
  printf("\n");

float h = 0.0001;
 float a = 2.0;
 float b = -3.0;
float c = 10.0;

float d1 = a*b +c;
c += h;
float d2 = a*b +c;
float d = (d2-d1)/h;
printf("d (slope) = %f\n", d);

  Value *w1 = store_value(2.0);
  print_value(w1);
  
  return 0;
}