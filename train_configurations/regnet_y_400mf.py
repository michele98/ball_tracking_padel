import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from utils.dataset import VideoDataset, MyConcatDataset
from utils.models import my_regnet_y_400mf
from utils.training import train_model


"============== Configure here =============="
class Config():
    # dataset config
    _sequence_length = 3
    _one_output_frame = True
    _image_size = (360, 640)

    # training
    _checkpoint_folder = './checkpoints/regnet_y_400mf_360_640'
    _batch_size = 16
    _epochs = 10

    def get_model(self):
        return my_regnet_y_400mf()

    def get_loss(self):
        return F.mse_loss

"============================================"


def target_transform(x):
    if x is None:
        x = [0, 0]
    return torch.tensor(x, dtype=torch.float32)


def create_datasets():
    """Create datasets with the right configuration"""
    config = Config()

    dataset_params = dict(image_size=config._image_size,
                          sequence_length=config._sequence_length,
                          output_heatmap=False,
                          one_output_frame=config._one_output_frame,
                          drop_duplicate_frames=True,
                          transform=ToTensor(),
                          target_transform=target_transform)

    roots = [f'../datasets/dataset_lluis/game{i+1}' for i in range(5)]

    # training dataset
    dataset_train_list = []
    dataset_train_list.append(VideoDataset(root="../datasets/dataset_finales_2020_en/", split='train', duplicate_equality_threshold=0.97, **dataset_params))
    for root in roots[:-1]:
        dataset_train_list.append(VideoDataset(root=root, duplicate_equality_threshold=1, **dataset_params))

    dataset_train = MyConcatDataset(dataset_train_list)

    # validation dataset
    dataset_val_list = []
    dataset_val_list.append(VideoDataset(root="../datasets/dataset_finales_2020_en/", split='val', duplicate_equality_threshold=0.97, **dataset_params))
    for root in roots[-1:]:
        dataset_val_list.append(VideoDataset(root=root, duplicate_equality_threshold=1, **dataset_params))

    dataset_val = MyConcatDataset(dataset_val_list)

    # test dataset (for now equal to validation dataset)
    dataset_test = dataset_val

    return dataset_train, dataset_val, dataset_test


def launch_training(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()

    dataset_train, dataset_val, dataset_test = create_datasets()
    data_loader_train = DataLoader(dataset_train, batch_size=config._batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=config._batch_size)

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

    dataset_train.save_info(os.path.join(config._checkpoint_folder, 'dataset_train_info.json'))
    dataset_val.save_info(os.path.join(config._checkpoint_folder, 'dataset_val_info.json'))
    dataset_test.save_info(os.path.join(config._checkpoint_folder, 'dataset_test_info.json'))


if __name__ == '__main__':
    launch_training()
