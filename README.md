## About 

Trying to implement [micrograd](https://github.com/karpathy/micrograd) [[yt]](https://youtu.be/VMj-3S1tku0) but in C.

Micrograd is a mini implementation for pytorch. Trying to do the implementation in C.

## Usage

```
// Initialize the values
Value* v1 = store_value(3.0);
Value* v2 = store_value(2.5);

// Perform the operation

// Value* res1 = sub(mul(add(v3, v4), v4), v3);    // (u+v)*v - u
// Value* res2 = relu(sub(v3, v4));
Value* result = power(v1, v2);
print_value(result);

// Perform backpropagation
backward(result);

// Print the gradients
print_value(v1);
print_value(v2);

// Free  memory
free_value(v1);
free_value(v2);
free_value(result);

```

