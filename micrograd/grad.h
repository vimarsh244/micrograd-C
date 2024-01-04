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


void back_add(Value* v);
void back_sub(Value* v);
void back_mul(Value* v);
void back_div(Value* v);
void back_power(Value* v);
void back_relu(Value* v);


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
    printf("Value: %f\t", v->data);
    printf("Gradient: %f\n", v->grad);
}

// store multiple values together
Value** store__multiple_values(float* arr){
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

Value* divide(Value* a, Value* b){
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
        
        for(int i=0; i<v->prev; i++){
            free_value(v->children[i]);
        }
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
    // printf("v->grad: %f\n", v->grad);
    // printf("v->children[0]->grad: %f\n", v->children[0]->grad);
    // printf("v->children[1]->grad: %f\n", v->children[1]->grad);

    // for single operation * is giving correct value, for multiple + is 
    // it was correct only i think just giving some pointers related issue cause multiple referencing of same value


    v->children[0]->grad += v->grad * v->children[1]->data;
    v->children[1]->grad += v->grad * v->children[0]->data;

    // printf("v->children[1]->grad: %f\n", v->children[1]->grad);
    // printf("v->children[0]->grad: %f\n", v->children[0]->grad);
}

//if y = (u_grad_prev, v_grad_prev => we take this into account) u/v
// dy/du = 1/v
// dy/dv = -u/v^2
void back_div(Value* v){
    v->children[0]->grad += v->grad / v->children[1]->data;
    v->children[1]->grad -= v->grad * (v->children[0]->data) / pow(v->children[1]->data, 2);
}

// y = u^v
// dy/du = v*u^(v-1)
// dy/dv = u^v * log(u)
void back_power(Value* v){

    v->children[0]->grad += v->grad * v->children[1]->data * pow(v->children[0]->data, v->children[1]->data - 1.0);
    if(v->children[0]->data > 0)
        v->children[1]->grad += v->grad * pow(v->children[0]->data, v->children[1]->data) * log(v->children[0]->data);
}


void back_relu(Value* v){
    if(v->children[0]->data > 0)
        v->children[0]->grad += v->grad;
    else 
        v->children[0]->grad += 0;
    
}

// topological sort
void build_map(Value* v, Value** map, int* map_size, Value** visited, int* visited_size){

    // if v is already in visited, return
    for(int i=0; i<*visited_size; i++){
        if(v == visited[i])
            return;
    }

    // if v is not in visited, add it to visited
    visited[*visited_size] = v;
    *visited_size += 1;

    // if v is a leaf node, add it to map
    // if(v->prev == 0){
    //     map[*map_size] = v;
    //     *map_size += 1;
    //     return;
    // }

    // if v is not a leaf node, build map for all children
    for(int i=0; i<v->prev; i++){
        build_map(v->children[i], map, map_size, visited, visited_size);
    }

    // add v to map => it should be leaf node if it arrives here
    map[*map_size] = v;
    *map_size += 1;
}

void backward(Value* root_v){

    //...
    // that bug was in cases where a node was used more than once,
    // if a variable is used more than once it causes errors
    // to fix it: we basically accumulate these gradients which explains it 
    // https://math.libretexts.org/Bookshelves/Calculus/Calculus_(OpenStax)/14%3A_Differentiation_of_Functions_of_Several_Variables/14.05%3A_The_Chain_Rule_for_Multivariable_Functions

    // need to watch video cant do it on my own
    float h = 0.00001;
    // We know dL/dd => but we need to find dL/da, dL/db, dL/dc
    // by product rule we know
    // dL/da = dL/dd * dd/da

    Value* map[100]; //sufficient ig
    int map_size = 0;
    Value* visited[100];
    int visited_size = 0;

    build_map(root_v, map, &map_size, visited, &visited_size);

    // printf("map size: %d\n",map_size);
    // printf("visited size: %d\n",visited_size);

    root_v->grad = 1.0;

    for(int i = map_size-1; i>=0; i--){
        // Value* v = map[i];
        // if(v->backward != NULL)
        //     printf("map %d value: %f\n", i, v->data);
        //     v->backward(v);

         if (map[i]->backward) {
            // printf("map %d value: %f\n", i, map[i]->data);
            
            map[i]->backward(map[i]);

            
        }
    }   
    
}

void back_exponentiate(Value* v){
    v->children[0]->grad += v->grad * exp(v->children[0]->data);
}

Value* exponentiate(Value* v){
    Value* res = (Value*) malloc(sizeof(Value));
    res->data = exp(v->data);
    res->grad = 0.0;
    res->children = (Value**) malloc(sizeof(Value*));
    res->children[0] = v;
    res->prev = 1;
    res -> backward = back_exponentiate;
    return res;
}

// d/dx tanh(x) = 1 - tanh(x)^2
void back_tanh(Value* v){
    v->children[0]->grad += v->grad * (1 - pow(tanh(v->children[0]->data), 2));
}

Value* def_tanh(Value* v){
    Value* out = (Value*) malloc(sizeof(Value));
    Value* v_neg_exp = store_value(-v->data);
    
    out->data = divide(sub(exponentiate(v),exponentiate(v_neg_exp)), add(exponentiate(v), exponentiate(v_neg_exp)))->data;
    out->grad = 0.0;
    out->children = (Value**) malloc(sizeof(Value*));
    out->children[0] = v;
    out->prev = 1;
    out->backward = back_tanh;
    free_value(v_neg_exp);
    return out;   
}

void back_log(Value* v){
    v->children[0]->grad += v->grad * (1/v->children[0]->data);
}

Value* def_log(Value* v){
    Value* out = (Value*) malloc(sizeof(Value));
    out->data = log(v->data);
    out->grad = 0.0;
    out->children = (Value**) malloc(sizeof(Value*));
    out->children[0] = v;
    out->prev = 1;
    out->backward = back_log;
    return out;
}

Value* def_sigmoid(Value* v){}
//todo

Value* def_softmax(Value* v){}