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

#define NUM_SAMPLES 50
#define NUM_EPOCHS 100
float learning_rate_prog = 0.0000001;

int main()
{
    srand(time(0));
    // Initialize the sizes of the layers in the MLP
    int sizes[] = {1, 2, 5, 1};
    int num_layers = sizeof(sizes) / sizeof(sizes[0]);

    // Initialize the MLP
    MLP *mlp = initialize_MLP(sizes, num_layers);

    // Create an artificial dataset of odd and even numbers
    Value *inputs[NUM_SAMPLES];
    Value *targets[NUM_SAMPLES];
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        int number = rand() % 100;
        inputs[i] = store_value(number);
        targets[i] = store_value(number % 2);
    }

    //print some random inputs and targets
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        printf("input: %f | target: %f\n", inputs[i]->data, targets[i]->data);
    }

    // Train the MLP
    // Train the MLP
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        for (int i = 0; i < NUM_SAMPLES; i++)
        {
            Value **input = &inputs[i];
            Value *target = targets[i];

            // Train the MLP on the current sample
            Value *loss = train_mlp(mlp, input, &target, learning_rate_prog);

            // Print the loss after every epoch
            // printf("Epoch %d Loss: %f\n", epoch, loss->data);
        }
    }

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
        int predicted_class = output[0]->data >= 0.5 ? 1 : 0;

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