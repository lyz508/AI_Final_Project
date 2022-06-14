import os

class ProjectConfig:
    def __init__(self, block_size, batch_size, buffer_size, data_name, epoch_times, **kwargs):
        self.block_size = block_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.data_name = data_name
        self.token_pos = f"tokenized_data/{data_name}"
        self.test_token_pos = f"tokenized_data/{data_name}-test"
        self.train_pos = f"trained_data/{data_name}/train.txt"
        self.test_pos = f"trained_data/{data_name}/test.txt"
        self.model_pos = f"trained_model/models/{data_name}"
        self.tfboard_pos = f"trained_model/logs/{data_name}"
        self.pltfigure_pos= f"trained_model/figure/{data_name}"
        self.epoch_times = epoch_times
        # Rest
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # Check whether the directories is existed
        # Check directory existed
        if not os.path.exists(self.token_pos):
            os.makedirs(self.token_pos)
        if not os.path.exists(self.train_pos):
            os.makedirs(self.train_pos)
        if not os.path.exists(self.model_pos):
            os.makedirs(self.model_pos)
        if not os.path.exists(self.tfboard_pos):
            os.makedirs(self.tfboard_pos)
        if not os.path.exists(self.pltfigure_pos):
            os.makedirs(self.pltfigure_pos)