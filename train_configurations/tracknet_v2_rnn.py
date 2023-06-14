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
from train_configurations.utils import collate_fn_rnn, BatchSamplerRNN
from train_configurations.utils import collate_fn_rnn, BatchSamplerRNN

"""Training schedule description

Definitions:
 - tc: total clearing (set all previous heatmaps to 0)
 - c:  clearing (set the previous input heatmaps to 0 from the first to the nth with a uniform probability distribution)
 - gt: use ground truth for the previous heatmaps
Note: gt is independent of tc and c. tc and c are not: if tc is True, c is automatically False.
We have to define:
 - p(tc)
 - p(c): to note, since we impose p(c|tc)=0, we have to sample p(c|not tc), for which we have p(c|not tc)=p(c)/(1-p(tc))
 - p(gt)

The training schedule is divided in 3 phases, each of which has 2 subphases:

Phase 1 (ground truth regime):
    - lr = 1e-3
    - Epochs 1-10:
        - p(tc) = 0.8
        - p(c)  = 0.1
        - p(gt) = 1
    - Epochs 11-15:
        - p(tc) = 0
        - p(c)  = 0.5
        - p(gt) = 0.8

Phase 2 (hybrid regime):
    - lr = 1e-3
    - Epochs 16-20:
        - p(tc) = 0
        - p(c)  = 1/3
        - p(gt) = 1/2
    - Epochs 21-25:
        - p(tc) = 0
        - p(c)  = 1/5
        - p(gt) = 1/5

Phase 3 (RNN regime):
    - lr = 1e-3
    - Epochs 26-30:
        - p(tc) = 0
        - p(c)  = 1/30
        - p(gt) = 1/30
    - Epochs 31-35:
        - p(tc) = 0
        - p(c)  = 1/200
        - p(gt) = 0

The idea is to make the model recognize the ball in the first phase,
and then to make it transition to tracking the ball from phase 2 to phase 3.
"""


"============== Configure here =============="
class Config():
    # dataset config
    _sequence_length = 4
    _one_output_frame = True
    _image_size = (360, 640)
    _grayscale = False

    # training
    _checkpoint_folder = './checkpoints/tracknet_v2_rnn_360_640'
    _batch_size =8

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
    # roots = [f'../datasets/prova' for _ in range(2)]

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

    if not os.path.exists(config._checkpoint_folder): os.makedirs(config._checkpoint_folder)
    dataset_train.save_info(os.path.join(config._checkpoint_folder, 'dataset_train_info.json'))
    dataset_val.save_info(os.path.join(config._checkpoint_folder, 'dataset_val_info.json'))
    dataset_test.save_info(os.path.join(config._checkpoint_folder, 'dataset_test_info.json'))

    data_loader_train = DataLoader(dataset_train, batch_sampler=BatchSamplerRNN(dataset_train, config._batch_size, drop_last=True))
    data_loader_val = DataLoader(dataset_val, batch_sampler=BatchSamplerRNN(dataset_val, config._batch_size, drop_last=True))

    model = config.get_model()

    phase_procedures = [
        {'epochs': 10, 'phase': 'phase_1_0', 'tc': 0.8, 'c': 0.1,   'gt': 1   },
        {'epochs': 5 , 'phase': 'phase_1_1', 'tc': 0,   'c': 0.5,   'gt': 0.8 },
        {'epochs': 5 , 'phase': 'phase_2_0', 'tc': 0,   'c': 1/3,   'gt': 1/2 },
        {'epochs': 5 , 'phase': 'phase_2_1', 'tc': 0,   'c': 1/5,   'gt': 1/5 },
        {'epochs': 5 , 'phase': 'phase_3_0', 'tc': 0,   'c': 1/30,  'gt': 1/30},
        {'epochs': 5 , 'phase': 'phase_3_1', 'tc': 0,   'c': 1/200, 'gt': 0   },
    ]

    for i, p in enumerate(phase_procedures):
        checkpoint_folder = os.path.join(config._checkpoint_folder, p['phase'])
        if not os.path.exists(checkpoint_folder): os.makedirs(checkpoint_folder)

        checkpoint_names = [name for name in os.listdir(checkpoint_folder) if '.ckpt' in name]

        # if there are no checkpoints in this phase folder, load last checkpoint from the previous phase
        if i>0 and len(checkpoint_names)==0:
            previous_checkpoints = [name for name in os.listdir(previous_checkpoint_folder) if '.ckpt' in name]
            model.load(os.path.join(previous_checkpoint_folder, previous_checkpoints[-1]))
        previous_checkpoint_folder = checkpoint_folder

        collate_fn = partial(collate_fn_rnn,
                            total_clear_probability=p['tc'],
                            clear_probability=p['c'],
                            ground_truth_probability=p['gt'],
                            sequence_length=config._sequence_length)

        data_loader_train.collate_fn = collate_fn
        data_loader_val.collate_fn = collate_fn
        train_model(model,
                    data_loader_train,
                    data_loader_val,
                    retain_graph=False,
                    loss_function=config.get_loss(),
                    epochs=p['epochs'],
                    resume_epoch_count_from_checkpoint=True,
                    checkpoint_folder=checkpoint_folder,
                    device=device,
                    additional_info={'dataset_train': dataset_train.get_info(),
                                     'dataset_val': dataset_val.get_info(),
                                     'dataset_test': dataset_test.get_info()})


if __name__ == '__main__':
    launch_training()
