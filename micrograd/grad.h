#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct Value{
    float data; //actual data of value stored
    float grad; // gradient wwrt
    struct Value** children; // children of the value
    int prev; // number of children
    void (*backward)(struct Value*); // function pointer to backward function

} Value;

// Initializizing a value
Value* value(float data){
    Value* v = (Value*) malloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->prev = 0;
    v->children = NULL;
    v->backward = NULL;
    return v;
}


void print_value(Value* v){
    printf("Value: %f\n", v->data);
    printf("Gradient: %f\n", v->grad);
}

