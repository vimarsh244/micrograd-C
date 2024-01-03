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
Value* store_value(float data){
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

// store multiple values together
void store__multiple_values(float* arr){
    int len = sizeof(arr)/sizeof(arr[0]);

    //malloc
    Value** v = (Value**) malloc(len*sizeof(Value*));

    if (v == NULL) { //borrowed code
        perror("Memory allocation failed");
        exit(1);
    }


    for(int i=0; i<len; i++){
        v[i] = store_value(arr[i]);
    }

    return v;
}

Value* add(Value* a, Value* b){
    Value* v = (Value*) malloc(sizeof(Value));
    v->data = a->data + b->data;
    v->grad = 0.0;
    v->children = (Value**) malloc(2*sizeof(Value*));
    v->children[0] = a;
    v->children[1] = b;
    v->prev = 2;

    v->backward = back_add;

    return v;
}

Value* sub(Value* a, Value* b){
    Value* v = (Value*) malloc(sizeof(Value));
    v->data = a->data - b->data;
    v->grad = 0.0;
    v->children = (Value**) malloc(2*sizeof(Value*));
    v->children[0] = a;
    v->children[1] = b;
    v->prev = 2;

    v->backward = back_sub;

    return v;
}

Value* mul(Value* a, Value* b){
    Value* v = (Value*) malloc(sizeof(Value));
    v->data = a->data * b->data;
    v->grad = 0.0;
    v->children = (Value**) malloc(2*sizeof(Value*));
    v->children[0] = a;
    v->children[1] = b;
    v->prev = 2;

    v->backward = back_mul;

    return v;
}

Value* div(Value* a, Value* b){
    Value* v = (Value*) malloc(sizeof(Value));
    v->data = a->data / b->data;
    v->grad = 0.0;
    v->children = (Value**) malloc(2*sizeof(Value*));
    v->children[0] = a;
    v->children[1] = b;
    v->prev = 2;

    v->backward = back_div;

    return v;
}

Value* power(Value* a, Value* b){
    Value* v = (Value*) malloc(sizeof(Value));
    v->data = pow(a->data, b->data);
    v->grad = 0.0;
    v->children = (Value**) malloc(2*sizeof(Value*));
    v->children[0] = a;
    v->children[1] = b;
    v->prev = 2;

    v->backward = back_power;

    return v;
}

Value* relu(Value* a){
    Value* v = (Value*) malloc(sizeof(Value));
    v->data = a->data > 0 ? a->data : 0;
    v->grad = 0.0;
    v->children = (Value**) malloc(sizeof(Value*));
    v->children[0] = a;
    v->prev = 1;

    v->backward = back_relu;

    return v;
}

void free_value(Value* v){
    
    if(v->prev > 0)
        free(v->children);
    
    free(v);
}


// y = v + u
// dy/dv = 1
// dy/du = 1
void back_add(Value* v){
    v->children[0]->grad += v->grad;
    v->children[1]->grad += v->grad;
    //to add gradient_clip functionality if i wish later
}

// y = v - u
// dy/dv = 1
// dy/du = -1
void back_sub(Value* v){
    v->children[0]->grad += v->grad;
    v->children[1]->grad -= v->grad;
}

// if y = vu
// dy/dv = u
// dy/du = v
// so below:
void back_mul(Value* v){
    v->children[0]->grad += v->grad * v->children[1]->data;
    v->children[1]->grad += v->grad * v->children[0]->data;
}

//if y = (u_grad_prev, v_grad_prev => we take this into account) u/v
// dy/du = 1/v
// dy/dv = -u/v^2
void back_div(Value* v){
    v->children[0]->grad += v->grad / v->children[1]->data;
    v->children[1]->grad += v->grad * -(v->children[0]->data) / pow(v->children[1]->data, 2);
}

// y = u^v
// dy/du = v*u^(v-1)
// dy/dv = u^v * log(u)
void back_power(Value* v){
    v->children[0]->grad += v->grad * v->children[1]->data * pow(v->children[0]->data, v->children[1]->data - 1);
    if(v->children[0]->data > 0)
        v->children[1]->grad += v->grad * pow(v->children[0]->data, v->children[1]->data) * log(v->children[0]->data);
}


void back_relu(Value* v){
    if(v->children[0]->data > 0)
        v->children[0]->grad += v->grad;
    else 
        v->children[0]->grad += 0;
    
}