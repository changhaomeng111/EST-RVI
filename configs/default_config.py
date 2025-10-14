class Config:
    # Training parameters
    learning_rate = 0.001
    training_epoch = 500
    units = 64
    seq_len = 4
    pre_len = 1  # Added for output_dim
    k = 8
    train_rate = 0.8
    batch_size = 32

    # Model parameters
    model_name = 'ST_AugGCN'
    lambda_loss = 0.0015

    # Data parameters
    num_nodes = 136  # Adjust according to actual data
    max_value = None  # Will be set at runtime

    # Path configurations
    data_path = './dataset/'
    output_path = './output/'

    # Corresponding to prediction length
    @property
    def output_dim(self):
        return self.pre_len