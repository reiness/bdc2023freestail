import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("Models", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.vocab = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.height = 64
        self.width = 256
        self.max_text_length = 9
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.train_epochs = 100
        self.train_workers = 8