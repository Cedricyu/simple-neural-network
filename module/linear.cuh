#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.cuh"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Linear {
    int in_features;
    int out_features;
    Tensor *weight;
    Tensor *bias;
} Linear;

Linear *linear_new(int in_f, int out_f);
void linear_free(Linear *self);
Tensor *linear_forward(Linear *self, Tensor *input);
Tensor *linear_tensor(Linear *self);
void linear_print_weight(Linear *self, const char *name);
void linear_print_grad(Linear *self, const char *name);

#ifdef __cplusplus
}
#endif

#endif
