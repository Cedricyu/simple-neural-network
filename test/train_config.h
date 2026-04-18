#ifndef TRAIN_CONFIG_H
#define TRAIN_CONFIG_H

typedef struct TrainingConfig {
    int input_dim;
    int hidden_dim1;
    int hidden_dim2;
    int hidden_dim3;
    int output_dim;
    int total_data;
    int batch_size;
    int epochs;
    int num_squares;
    float learning_rate;
    unsigned int seed;
} TrainingConfig;

TrainingConfig training_config_default(void);

#endif
