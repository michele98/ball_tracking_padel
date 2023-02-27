import os
import json
import time
import numpy as np
import pandas as pd

import torch
import torch.utils.data

from utils.dataset import VideoDataset, MyConcatDataset

from typing import Union


def get_maximum_coordinates(heatmaps):
    y, x = np.nonzero(heatmaps == np.max(heatmaps))
    return x[0], y[0]


def compute_positions(net : torch.nn.Module,
                      dataloader : torch.utils.data.DataLoader,
                      device : str = 'cpu',
                      prefix : str = ''):
    """Compute the position of the ball for all the frames in the dataloader.

    Parameters
    ----------
    net : torch.nn.Module
        the model to evaluate
    dataloader : torch.utils.data.DataLoader
        dataloader of the test set
    device : torch.device, optional
        cpu or cuda, by default cpu.

    Returns
    -------
    true_positions : np.ndarray
        list of ground truth positions in (y, x) pixel coordinates
    predicted_positions : np.ndarray
        list of predicted positions in (y, x) pixel coordinates
    """

    net.eval() # set the model in evaluation mode

    true_positions = []
    predicted_positions = []

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            inputs = data[0].to(device)
            labels = data[1]

            outputs = net(inputs).to('cpu')

            for label, output in zip(labels, outputs):
                true_positions.append(get_maximum_coordinates(label[-1].numpy()))
                predicted_positions.append(get_maximum_coordinates(output[-1].numpy()))

            epoch_time = time.time() - start_time
            batch_time = epoch_time/(batch_idx+1)
            print(prefix + f'{batch_idx+1}/{len(dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step'.ljust(80), end = '\r')
    print(prefix + f'{batch_idx+1}/{len(dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step'.ljust(80))

    return np.array(true_positions), np.array(predicted_positions)


# TODO: finish this
# TODO: write docstring
def save_outputs(true_positions : Union[list, np.ndarray],
                 predicted_positions : Union[list, np.ndarray],
                 dataset : Union[VideoDataset, MyConcatDataset] = None,
                 output_folder : str = './',
                 output_filename : str = 'output.csv'):

    image_size = (1, 1)
    dataset_info = [{'image_size': image_size, 'len': len(predicted_positions)}]

    if dataset is not None:
        dataset_info = dataset.get_info()
        if type(dataset) is VideoDataset:
            dataset_info = [dataset_info]

        with open(os.path.join(output_folder, 'output_info.json'), 'w') as f:
            json.dump(dataset_info, f, default=str, indent=2)

        image_size = dataset.get_info()[0]['image_size']

    # TODO: finish this
    # dataset_ids = []
    # frame_number = []
    # for dataset_id, info in enumerate(dataset_info):
    #     dataset_ids += [dataset_id for i in range(dataset_info['len'])]
    #     df_dataset = pd.read_csv('')
    #     frame_numbers += [0]

    # reminder: image_size has (height, width)
    df_out = pd.DataFrame({'x_true': true_positions[:,0]/image_size[1],
                           'y_true': true_positions[:,1]/image_size[0],
                           'x_pred': predicted_positions[:,0]/image_size[1],
                           'y_pred': predicted_positions[:,1]/image_size[0]})

    df_out.to_csv(os.path.join(output_folder, output_filename), index=False)
