import tensorflow as tf

class ProjectConfig:
    def __init__(self, block_size, batch_size, buffer_size, data_name, epoch_times, **kwargs):
        self.block_size = block_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.data_name = data_name
        self.token_pos = f"tokenized_data/{data_name}"
        self.train_pos = f"trained_data/{data_name}/train.txt"
        self.model_pos = f"trained_model/models/{data_name}"
        self.tfboard_pos = f"trained_model/logs/{data_name}"
        self.epoch_times = epoch_times
        # Rest
        for k, v in kwargs.items():
            setattr(self, k, v)
