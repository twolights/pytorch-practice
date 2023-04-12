from .preprocess import image_to_tensor, CatsDogsDataset
from torch.utils.data import random_split, DataLoader
from . import config


all_training_set = CatsDogsDataset(config.TRAINING_SET_DIR, is_training=True)
training_set, validation_set = random_split(all_training_set, [0.8, 0.2])
testing_set = CatsDogsDataset(config.TESTING_SET_DIR, is_training=False)

training_dataloader = DataLoader(training_set, batch_size=config.BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(training_set, batch_size=config.BATCH_SIZE, shuffle=True)
testing_dataloader = DataLoader(training_set, batch_size=config.BATCH_SIZE, shuffle=True)
