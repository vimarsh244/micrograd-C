#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "grad.h"

int main() {
    // Initialize the values
    Value* v1 = store_value(3.0);
    Value* v2 = store_value(2.5);

    // Perform the operation
    Value* result = power(v1, v2);

    // Perform backpropagation
    backward(result);

    // Print the gradients
    // print_value(v1);
    // print_value(v2);


    Value* v3 = store_value(3.0); // u
    Value* v4 = store_value(4.0); //v
    Value* res3 = sub(mul(add(v3, v4), v4), v3); // (u+v)*v - u
    // Value* res2 = relu(sub(v3, v4));
    // Value* res3 = power(v3, v4);

    //dres/du
    // v -1
    //dres/dv
    // u + 2v
    float u = 3.0, v = 4.0;
    printf("dres/du: %f\n", v-1);
    printf("dres/dv: %f\n", u+2*v);

    // printf("res->value: %f\n", res->data);

    // backward(res);
    // print_value(res);
    // // backward(res2);
    // print_value(v3);
    // print_value(v4);

    // y= u^v
    // dy/du = v*u^(v-1)
    // dy/dv = u^v * ln(u)
    // printf("dres/du: %f\n", v*pow(u, v-1));
    // printf("dres/dv: %f\n", pow(u, v)*log(u));
    backward(res3);
    print_value(res3);
    print_value(v3);
    print_value(v4);
    
    // Free the memory
    free_value(v3);
    free_value(v4);
    free_value(res3);

    return 0;
}