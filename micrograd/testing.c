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


    Value* v3 = store_value(3.0);
    Value* v4 = store_value(4.0);
    Value* res = sub(mul(add(v3, v4), v4), v3);
    Value* res2 = add(v3, v4);
    Value* res3 = sub(v3, v4);

    // backward(res);
    print_value(res);
    backward(res2);
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