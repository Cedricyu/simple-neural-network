#include "train_config.h"

#include <time.h>

TrainingConfig training_config_default(void) {
    TrainingConfig cfg;

    cfg.input_dim = 2;
    cfg.hidden_dim1 = 8;
    cfg.hidden_dim2 = 16;
    cfg.hidden_dim3 = 8;
    cfg.output_dim = 2;
    cfg.total_data = 1024;
    cfg.batch_size = 32;
    cfg.epochs = 50;
    cfg.num_squares = 2;
    cfg.learning_rate = 0.001f;
    cfg.seed = (unsigned int)time(NULL);

    return cfg;
}
