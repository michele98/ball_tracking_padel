import os
import numpy as np
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from detection.dataset import VideoDatasetRNN, MyConcatDataset
from detection.models import TrackNetV2RNN
from detection.training import train_model
from train_configurations.utils import collate_fn_rnn


"============== Configure here =============="
class Config():
    # dataset config
    _sequence_length = 4
    _one_output_frame = True
    _image_size = (360, 640)
    _grayscale = False

    # training
    _checkpoint_folder = './checkpoints/tracknet_v2_rnn_360_640_gt'
    _batch_size = 1
    _epochs = 20
    _clear_probability = 0.3
    _total_clear_probability = 0
    _ground_truth_probability = 1

    def get_model(self):
        return TrackNetV2RNN(sequence_length=self._sequence_length, one_output_frame=self._one_output_frame, grayscale=self._grayscale)

    def get_loss(self):
        return F.mse_loss

"============================================"


def create_datasets():
    """Create datasets with the right configuration"""
    config = Config()

    dataset_params = dict(image_size=config._image_size,
                          sigma=5,
                          sequence_length=config._sequence_length,
                          one_output_frame=config._one_output_frame,
                          duplicate_equality_threshold=0.97,
                          transform=ToTensor(),
                          target_transform=ToTensor(),
                          grayscale=config._grayscale)

    roots = [f'../datasets/dataset_lluis/game{i+1}' for i in range(5)]
    # roots = [f'../datasets/dataset_lluis/game{i}' for i in [2, 1]]
    # roots = [f'../datasets/prova_2' for _ in range(2)]

    # training dataset
    dataset_train_list = []
    dataset_train_list.append(VideoDatasetRNN(root="../datasets/dataset_finales_2020_en/", split='train', drop_duplicate_frames=True, **dataset_params))
    for root in roots[:-1]:
        dataset_train_list.append(VideoDatasetRNN(root=root, drop_duplicate_frames=False, **dataset_params))

    dataset_train = MyConcatDataset(dataset_train_list)

    # validation dataset
    dataset_val_list = []
    dataset_val_list.append(VideoDatasetRNN(root="../datasets/dataset_finales_2020_en/", split='val', drop_duplicate_frames=True, **dataset_params))
    for root in roots[-1:]:
        dataset_val_list.append(VideoDatasetRNN(root=root, drop_duplicate_frames=False, **dataset_params))

    dataset_val = MyConcatDataset(dataset_val_list)

    # test dataset (for now equal to validation dataset)
    dataset_test = dataset_val

    return dataset_train, dataset_val, dataset_test


def launch_training(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()

    dataset_train, dataset_val, dataset_test = create_datasets()

    collate_fn = partial(collate_fn_rnn,
                        total_clear_probability=config._total_clear_probability,
                        clear_probability=config._clear_probability,
                        ground_truth_probability=config._ground_truth_probability,
                        sequence_length=config._sequence_length)

    data_loader_train = DataLoader(dataset_train, batch_size=config._batch_size, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=config._batch_size, collate_fn=collate_fn)

    if not os.path.exists(config._checkpoint_folder): os.makedirs(config._checkpoint_folder)

    dataset_train.save_info(os.path.join(config._checkpoint_folder, 'dataset_train_info.json'))
    dataset_val.save_info(os.path.join(config._checkpoint_folder, 'dataset_val_info.json'))
    dataset_test.save_info(os.path.join(config._checkpoint_folder, 'dataset_test_info.json'))

    train_model(config.get_model(),
                data_loader_train,
                data_loader_val,
                loss_function=config.get_loss(),
                epochs=config._epochs,
                checkpoint_folder=config._checkpoint_folder,
                device=device,
                additional_info={'dataset_train': dataset_train.get_info(),
                                 'dataset_val': dataset_val.get_info(),
                                 'dataset_test': dataset_test.get_info()})


if __name__ == '__main__':
    launch_training()
