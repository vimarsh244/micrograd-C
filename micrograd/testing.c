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
    print_value(v1);
    print_value(v2);


    Value* v3 = store_value(3.0); // u
    Value* v4 = store_value(4.0); //v
    Value* res = sub(power(add(v3, v4), v4), v3); // (u+v)^v - u
    Value* res2 = relu(sub(v3, v4));
    Value* res3 = divide(v3, v4);

    //dres/du
    // v*(u+v)^(v-1) - 1
    // dres/dv
    // u*(u+v)^(v-1) + (u+v)^v*log(u+v) - u

    float u = 3.0, v = 4.0;
    printf("value of dres/du: %f\n", v*pow(u+v, v-1) - 1.0);
    printf("value of dres/dv: %f\n", u*pow(u+v, v-1) + pow(u+v, v)*log(u+v) - u);
    printf("res->value: %f\n", res->data);

    backward(res);
    print_value(res);
    // backward(res2);
    print_value(v3);
    print_value(v4);

    backward(res3);
    print_value(v3);
    print_value(v4);
    
    // Free the memory
    free_value(v1);
    free_value(v2);
    free_value(result);

    return 0;
}