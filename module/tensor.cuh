// module/Tensor.h (C-style, no STL)
#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "float_precision.h"

typedef struct Tensor Tensor;
typedef struct Dependency Dependency;

typedef Tensor *(*BackwardFn)(Tensor *a, Tensor *b, Tensor *grad_out);

struct Dependency {
    Tensor *tensor;
    Tensor *ctx;
    BackwardFn backward_fn;
};

struct Tensor {
    // Device pointers (GPU memory)
    float_t *data;
    float_t *grad;

    int *shape;
    int ndim;
    int requires_grad;
    int owns_data;
    int owns_grad;

    Dependency *deps;
    int num_deps;

    // Optional owner-defined config (e.g. conv stride/padding) that a layer
    // attaches to its own weight tensor so its backward_fn can read it back
    // via the `a`/`ctx` tensor it already receives. Freed by tensor_free.
    void *meta;

    // Momentum buffer for tensor_update_momentum (device memory, lazily allocated, freed by tensor_free).
    float_t *velocity;
};

// Device memory pool. Every op allocates its output (and every backward its gradient) afresh each
// step, and the sizes repeat step after step, so freed buffers are kept on a list and handed back
// for the next request of exactly the same size instead of going through cudaMalloc/cudaFree
// (which synchronize the device). Returned memory is NOT zeroed.
void *tensor_pool_alloc(size_t bytes);
void tensor_pool_free(void *ptr, size_t bytes);
// Gives every cached buffer back to the driver (call when a model is thrown away).
void tensor_pool_release(void);

Tensor *tensor_create(int ndim, int *shape, int requires_grad);
Tensor *tensor_from_data(float *host_data, int ndim, int *shape);
Tensor *tensor_from_data_2d(float *host_data, int dim0, int dim1);
Tensor *tensor_from_device(float *device_data, int ndim, int *shape, int requires_grad);
int tensor_numel(int ndim, int *shape);
void tensor_to_host(Tensor *t, float *host_out);
void tensor_set_data(Tensor *t, const float *host_data);      // overwrite the values from host memory
void tensor_grad_to_host(Tensor *t, float *host_out);         // copy the accumulated gradient (zeros if none)
void tensor_zero_grad(Tensor *t);
void tensor_add_dependency(Tensor *t, Tensor *dep_tensor, BackwardFn fn);
void tensor_add_dependency_with_ctx(Tensor *t, Tensor *dep_tensor, Tensor *ctx, BackwardFn fn);
void tensor_backward(Tensor *t, Tensor *grad_output);
void tensor_free(Tensor *t);
void tensor_print(Tensor *t);
void tensor_print_grad(Tensor *t);
void tensor_grad(Tensor *t, Tensor *grad);
void tensor_update(Tensor *t, float lr);
// SGD with classical momentum: v = momentum*v + grad; w -= lr*v. momentum <= 0 is plain tensor_update.
void tensor_update_momentum(Tensor *t, float lr, float momentum);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_H
