#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "grad.h" // dont do this

#include <time.h> // for seeding random number generator


/*
Need to think to know how to structure it

What kind of NN I want?

1 Neuron:
 - n inputs
    to w_i*x_i + b
apply activation at neuron level or layer level?

[
    Neuron1 Neuron2 Neuron3 ...
] Layer 1
... Layer N
*/

void show_params();

typedef enum {
    NO,
    RELU,
    TANH,
    SIGMOID,
    SOFTMAX
} act_fn;

typedef struct Neuron{
    Value** weights;
    Value* bias;
    int input_size;
    
    // https://www.geeksforgeeks.org/enumeration-enum-c/ 
    
    act_fn activation_function;
} Neuron;



Neuron* initialize_Neuron(int input_size, act_fn activation_function){
    Neuron* n = (Neuron*) malloc(sizeof(Neuron));
    n->input_size = input_size;
    n->activation_function = activation_function;
    n->weights = (Value**) malloc(input_size*sizeof(Value*));
    for(int i = 0; i< input_size; i++){

        // n->weights[i] = store_value(rand()*2.0 - 1.0); // THIS IS WRNG random number between -1 and 1
        // i want to generate values between -1 and 1 how to do it:
        n->weights[i] = store_value(((float)rand()/(float)(RAND_MAX)) * 2.0 - 1.0); // it was generating same random valuesss whyyyy = we seed it

        printf("%f\n", n->weights[i]->data);
    }
    n->bias = store_value(0.0);
    return n;
}



Value* activation_output(Value* v, act_fn activation_function){
    // printf("activation function: %d\n", activation_function);
    Value* out = store_value(0.0);
    switch (activation_function){
        case NO:
            out = v;
            break;
        case RELU:
            out = relu(v);
            break;
        case TANH:
            out = def_tanh(v);
            break;
        case SIGMOID:
            out= def_sigmoid(v);
            break;
        case SOFTMAX:   
            out = def_softmax(v);
            break;
    }
    return out;
}

Value* forward_pass_Neuron(Neuron* n, Value** inputs){
    Value* sum = store_value(0.0);
    for(int i = 0; i < n->input_size; i++){
        // Value* prod = mul(n->weights[i], inputs[i]);
        // sum = add(sum, prod);
        // // free_value(prod);

        sum = add(sum, mul(n->weights[i], inputs[i]));
        // sum = summation over i (w_i * x_i)
    }
    sum = add(sum, n->bias); // adding bias term
    // printf("activation function: %d\n", n->activation_function);
    sum = activation_output(sum, n->activation_function);
    //change of plans: activation function after a layer not after a neuron

    return sum;
}

// layer is just many neurons stacked so

typedef struct Layer{
    Neuron** neurons;
    // int input_size; 
    int output_size;
} Layer;

Layer* inititalize_Layer(int input_size, int output_size, act_fn activation_function){
    Layer* l = (Layer*) malloc(sizeof(Layer));
    // l->input_size = input_size;
    printf("input size: %d | output size: %d\n", input_size, output_size);
    l->output_size = output_size;
    l->neurons = (Neuron**) malloc(output_size*sizeof(Neuron*));
    for(int i = 0; i< output_size; i++){
        l->neurons[i] = initialize_Neuron(input_size, activation_function);
    }
    return l;
}

Value** forward_pass_Layer(Layer* l, Value** inputs){
    Value** outputs = (Value**) malloc(l->output_size*sizeof(Value*));
    for(int i = 0; i< l->output_size; i++){
        outputs[i] = forward_pass_Neuron(l->neurons[i], inputs);
    }
    return outputs;
}


// multi layer perceptron
typedef struct{
    Layer** layers;
    int num_layers;
} MLP;

// sizes = [2,3,2] | num_layers = 3
MLP* initialize_old_MLP(int* sizes, int num_layers){
    MLP* mlp = (MLP*) malloc(sizeof(MLP));
    mlp->num_layers = num_layers -1; // not -1 : so implement accordingly 
    // printf("number of layers: %d\n", num_layers);   
    mlp->layers = (Layer**) malloc((num_layers-1)*sizeof(Layer*));
    for(int i = 0; i< num_layers - 1; i++){
        mlp->layers[i] = inititalize_Layer(sizes[i], sizes[i+1], NO);
        printf("%d\n", i);
    }
    act_fn activation_function = TANH;
    // printf("activatassadaion function: %d\n", activation_function);
    // printf("reached here\n");
    mlp->layers[num_layers-1] = inititalize_Layer(sizes[num_layers-1], 1, activation_function); // last layer has an activation function
    // printf("can u reach here\n");
    // show_params(mlp);
    return mlp;
}

MLP* initialize_MLP(int* sizes, int nlayers) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    mlp->layers = (Layer**)malloc((nlayers - 1) * sizeof(Layer*));
    for (int i = 0; i < nlayers - 1; i++) {
        int av= TANH;
        if(i != nlayers - 2)
            av = NO;  // nonlinearity for all layers except the last one

        mlp->layers[i] = inititalize_Layer(sizes[i], sizes[i+1], av);
    }
    mlp->num_layers = nlayers - 1;
    return mlp;
}


// Value** forward_pass_MLP(MLP* mlp, Value** inputs){
//     // printf("reached here\n");
//     Value** outputs = (Value**) malloc(mlp->num_layers*sizeof(Value*));
//     for(int i = 0; i< mlp->num_layers; i++){
//         outputs = forward_pass_Layer(mlp->layers[i], inputs);
//         // printf("reached here\n");
//     }
//     return outputs;
// }

Value** forward_pass_MLP(MLP* mlp, Value** x) {
    for (int i = 0; i < mlp->num_layers; i++) {
        x = forward_pass_Layer(mlp->layers[i], x);
    }
    return x;
}

// Value** forward_pass_MLP(MLP* mlp, Value** inputs){
//     Value** outputs = (Value**) malloc(mlp->num_layers*sizeof(Value*));
//     // printf("mlp num layers: %d\n", mlp->num_layers);
//     outputs = forward_pass_Layer(mlp->layers[0], inputs);
//     for(int i = 1; i< mlp->num_layers; i++){
//         Value** temp = forward_pass_Layer(mlp->layers[i], outputs);
//         if(i > 0) free(outputs); // Free the previous outputs
//         outputs = temp;
//     }
//     print_value(outputs[0]);
//     return outputs;
// }

Value* mse_loss(Value** y_predicted, Value** y_true, int size){
    Value* loss = store_value(0.0);
    for(int i = 0; i< size; i++){
        // print_value(sub(y_predicted[i], y_true[i])); why is y_predicted turning out to be 0
        loss = add(loss, power(sub(y_predicted[i], y_true[i]), store_value(2.0)));
    }
    loss = divide(loss, store_value(size));
    // print_value(loss);
    return loss;
}

Value* cross_entropy_loss(Value** y_predicted, Value** y_true, int size){
    Value* loss = store_value(0.0);
    for(int i = 0; i< size; i++){
        loss = add(loss, mul(y_true[i], def_log(y_predicted[i])));
    }
    loss = divide(loss, store_value(size));
    return loss;
}


void update_weights(Value* v, float learning_rate){
    v->data = v->data - learning_rate*v->grad;
}

//copy pasta
void show_params(MLP* mlp){
    printf("\nMLP\n");
    for (int i = 0; i < mlp->num_layers; i++) {
        Layer* layer = mlp->layers[i];
        printf("\nLayer%i:\n", i);
        for (int j = 0; j < layer->output_size; j++) {
            Neuron* neuron = layer->neurons[j];
            for (int k = 0; k < neuron->input_size; k++) {
                print_value(neuron->weights[k]);
            }
        }
    }
        printf("\n\n");
}



// dont fill HEAP and crash like u did

void free_neuron(Neuron* n){
    for(int i = 0; i< n->input_size; i++){
        free_value(n->weights[i]);
    }
    free_value(n->bias);
    free(n->weights);
    free(n);
}

void free_layer(Layer* l){
    for(int i = 0; i< l->output_size; i++){
        free_neuron(l->neurons[i]);
    }
    free(l->neurons);
    free(l);
}

void free_MLP(MLP* mlp){
    for(int i = 0; i< mlp->num_layers; i++){
        free_layer(mlp->layers[i]);
    }
    free(mlp->layers);
    free(mlp);
}


Value* train_mlp(MLP* mlp, Value** x, Value** y_actual, float learning_rate){
    // print_value(x[0]);
    // print_value(y_actual[0]);
    Value** y_predicted = forward_pass_MLP(mlp, x);
    // print_value(y_predicted[0]);

    // printf("mlp->layers[mlp->num_layers-1]->output_size: %d\n", mlp->layers[mlp->num_layers-1]->output_size);
    Value* loss = mse_loss(y_predicted, y_actual, mlp->layers[mlp->num_layers-1]->output_size);
    // printf("loss: %f\n", loss->data);
    for(int i = 0; i< mlp->num_layers; i++){
        Layer* layer = mlp->layers[i];
        for(int j = 0; j< layer->output_size; j++){
            Neuron* neuron = layer->neurons[j];            
            update_weights(neuron->bias, learning_rate);
            for(int k = 0; k< neuron->input_size; k++){
                update_weights(neuron->weights[k], learning_rate);
            }
        }
    }
    return loss;
    free(y_predicted);
}