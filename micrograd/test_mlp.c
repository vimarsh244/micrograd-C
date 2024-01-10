// // #include "grad.h"
// #include "neuron.h"
// #include <stdio.h>

// int main() {
//     // Initialize the sizes of the layers in the MLP
//     int sizes[] = {2, 3, 1};
//     int num_layers = sizeof(sizes) / sizeof(sizes[0]);

//     // Initialize the MLP
//     MLP* mlp = initialize_MLP(sizes, num_layers);

//     // Create some dummy input data
//     Value* input1 = store_value(0.5);
//     Value* input2 = store_value(-0.5);
//     Value* inputs[] = {input1, input2};

//     // Perform a forward pass through the MLP
//     Value** outputs = forward_pass_MLP(mlp, inputs);

//     // Print the output of the MLP
//     for (int i = 0; i < mlp->layers[num_layers-1]->output_size; i++) {
//         printf("Output %d: %f\n", i, outputs[i]->data);
//     }

//     // Free the MLP
//     free_MLP(mlp);

//     return 0;
// }

// #include "grad.h"
#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>

#define NUM_SAMPLES 10
#define NUM_EPOCHS 10
float learning_rate_prog = 0.001;


//one hot encoding values
float* one_hot_encode(float y_true) {
    float* encoded = (float*)malloc(2 * sizeof(float));
    if (y_true==1.0){
        encoded[0] = 0.0;
        encoded[1] = 1.0;
    }
    else{
        encoded[0] = 1.0;
        encoded[1] = 0.0;
    }
    return encoded;
}

int main()
{
    srand(time(0));
    // printf("%f\n", ((float)rand()/(float)(RAND_MAX)) * 2.0 - 1.0);
    // Initialize the sizes of the layers in the MLP
    int sizes[] = {1, 5, 2};
    int num_layers = sizeof(sizes) / sizeof(int);

    // Initialize the MLP
    MLP *mlp = initialize_MLP(sizes, num_layers);

    // Create an artificial dataset of odd and even numbers
    Value** inputs = (Value**) malloc(NUM_SAMPLES*sizeof(Value*));
    Value** targets = (Value**) malloc(NUM_SAMPLES*sizeof(Value*));
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        int number = rand() % 100;
        inputs[i] = store_value(number);
        if(number % 2 == 0){
            targets[i] = store_value(1.0);
        }
        else
            targets[i] = store_value(-1.0);
        // targets[i] = store_value(number % 2);
    }

    //print some random inputs and targets
    // Print the inputs and targets
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        //printf("i: %d, input: %f | target: %f\n", i, inputs[i]->data, targets[i]->data);
        printf("bhai bt");
    }
    printf("ayo wat");

    // Train the MLP new

    Value* total_loss = store_value(0.0);
    float epoch_loss = 0.0;
    int backward_freq = 2;

    for (int ep = 0; ep < NUM_EPOCHS; ep++) {
        for (int i=0; i < NUM_SAMPLES; i++) {
            
            Value** x = &inputs[i];
            // printf("input: %f\n", x[0]->data);

            float* arr_y = one_hot_encode(targets[i]->data);
            Value** y_true = store__multiple_values(arr_y);

            Value* loss = train_mlp(mlp, x, y_true, learning_rate_prog);
            total_loss = add(total_loss, loss);
            epoch_loss+=total_loss->data;

            // Backward pass
            if (i%backward_freq==0){
                // make loss.grad=1.0 (last node in mlp topo graph).
                // grad is basically dy/da or dy/db where y = a op b; op can by anything add, sub, div ..
                // in case of last node (which is the loss) -> a or b is itself y. since its the last node, and does have any op on it.
                // so basically dy/dy = 1.0 
                // This kicks off the gradient propogation backwards.
                total_loss->grad=1.0;
                backward(total_loss);
                // resetting total_loss for a new epoch.
                // free_value(total_loss);
                total_loss = store_value(0.0);
            }
            
            
            // show_params(mlp);
        }
        printf("\n\nEPOCH %i LOSSS: %f", ep, epoch_loss/25);
        epoch_loss=0.0;
        
    }

    /*
    // Train the MLP
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        for (int i = 0; i < NUM_SAMPLES; i++)
        {
            Value **input = &inputs[i];
            Value *target = targets[i];

            // Train the MLP on the current sample
            Value *loss = train_mlp(mlp, input, &target, learning_rate_prog);

            
            backward(loss);
            // Print the loss after every epoch
            printf("Epoch %d Loss: %f\n", epoch, loss->data);
        }
    }
    */
    printf("finifhsed training\n");
    show_params(mlp);
    // Test the MLP
    int num_correct = 0;
    for (int i = 0; i < NUM_SAMPLES; i++)
    {

        // printf("entered for loop itr: %d\n", i);
        int number = rand() % 100;
        Value *input = store_value(number);

        printf("input: %f\n", input->data);
        // generate me two inputs that i can pass together
        // printf("brother");

        Value *inputs[] = {input}; // Create an array of inputs
        // printf("brother");
        Value **output = forward_pass_MLP(mlp, inputs);

        // printf("size of output: %ld\n", sizeof(output) / sizeof(output[0]));

        printf("output: %f\n", output[0]->data);

        // Round the output to get the predicted class
        int predicted_class = output[0]->data >= 0 ? 1 : 0;

        // Check if the prediction is correct
        if (predicted_class == number % 2)
        {
            num_correct++;
        }
    }

    // Print the accuracy of the MLP
    printf("Accuracy: %f\n", (float)num_correct / NUM_SAMPLES);

    // Free the MLP
    free_MLP(mlp);

    return 0;
}