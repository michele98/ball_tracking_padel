import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from detection.dataset import VideoDataset, MyConcatDataset
from detection.models import TrackNetV2MSE
from detection.training import train_model


"============== Configure here =============="
class Config():
    # dataset config
    _sequence_length = 3
    _one_output_frame = True
    _image_size = (360, 640)

    # training
    _checkpoint_folder = './checkpoints/tracknet_v2_mse_360_640'
    _batch_size = 2
    _epochs = 10

    def get_model(self):
        return TrackNetV2MSE(sequence_length=self._sequence_length, one_output_frame=self._one_output_frame)

    def get_loss(self):
        return F.mse_loss

"============================================"


def create_datasets():
    """Create datasets with the right configuration"""
    config = Config()

    dataset_params = dict(image_size=config._image_size,
                          sigma=5,
                          sequence_length=config._sequence_length,
                          heatmap_mode='image',
                          one_output_frame=config._one_output_frame,
                          duplicate_equality_threshold=0.97,
                          transform=ToTensor(),
                          target_transform=ToTensor())

    roots = [f'../datasets/dataset_lluis/game{i+1}' for i in range(5)]

    # training dataset
    dataset_train_list = []
    dataset_train_list.append(VideoDataset(root="../datasets/dataset_finales_2020_en/", split='train', drop_duplicate_frames=True, **dataset_params))
    for root in roots[:-1]:
        dataset_train_list.append(VideoDataset(root=root, drop_duplicate_frames=False, **dataset_params))

    dataset_train = MyConcatDataset(dataset_train_list)

    # validation dataset
    dataset_val_list = []
    dataset_val_list.append(VideoDataset(root="../datasets/dataset_finales_2020_en/", split='val', drop_duplicate_frames=True, **dataset_params))
    for root in roots[-1:]:
        dataset_val_list.append(VideoDataset(root=root, drop_duplicate_frames=False, **dataset_params))

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
