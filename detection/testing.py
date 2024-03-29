import os
import cv2
import json
import time
import shutil
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from torch.utils.data import DataLoader

from train_configurations.utils import get_standard_test_dataset, get_debug_dataset

from typing import Union


def get_global_maximum(heatmaps):
    y, x = np.nonzero(heatmaps == np.max(heatmaps))
    return x[0], y[0]


def get_local_maxima(array, include_diagonal: bool = True, threshold: float = 0, sigma: float = 0):
    """Find local maxima in a 2D array by comparing pixels in the immediate neighborhood

    Parameters
    ----------
    array : 2D np.ndarray
    include_diagonal : bool, optional
        if True, pixels in the diagonal are included in the immediate neighborhood. By default True
    threshold : int, optional
        only maxima above this value will be considered. By default 0
    sigma : int, optional
        if provided, applies gaussian blurring to the image before computing the maxima

    Returns
    -------
    ndarray
        postions of the local maxima. The shape is `(n_maxima, 2)`
    """
    padded_array = array.copy()

    # smooth image for removing neighboring maxima
    if sigma>0:
        # save value range for later rescaling back
        max_value = padded_array.max()
        min_value = padded_array.min()

        # apply gaussian blur
        ks = int(4*2*np.ceil(sigma)+1)
        padded_array = cv2.GaussianBlur(padded_array, (ks, ks), sigma)

        # rescale blurred array to have the same range as before
        denominator = padded_array.max() - padded_array.min()
        denominator = 1 if denominator==0 else denominator
        a = (max_value - min_value) / denominator
        b = max_value - a*padded_array.max()
        padded_array = a*padded_array + b

    # Pad array so we can handle edges
    padded_array = np.pad(padded_array, ((1,1),(1,1)), constant_values=threshold)

    # Determine if each location is bigger than adjacent neighbors
    adjacentmax = (
    (padded_array[1:-1,1:-1] > threshold) &
    (padded_array[0:-2,1:-1] <= padded_array[1:-1,1:-1]) &
    (padded_array[2:,  1:-1] <= padded_array[1:-1,1:-1]) &
    (padded_array[1:-1,0:-2] <= padded_array[1:-1,1:-1]) &
    (padded_array[1:-1,2:  ] <= padded_array[1:-1,1:-1])
    )
    if not include_diagonal :
        return np.argwhere(adjacentmax)

    # Determine if each location is bigger than diagonal neighbors
    diagonalmax = (
    (padded_array[0:-2,0:-2] <= padded_array[1:-1,1:-1]) &
    (padded_array[2:  ,2:  ] <= padded_array[1:-1,1:-1]) &
    (padded_array[0:-2,2:  ] <= padded_array[1:-1,1:-1]) &
    (padded_array[2:  ,0:-2] <= padded_array[1:-1,1:-1])
    )

    return np.argwhere(adjacentmax & diagonalmax)


def _get_checkpoint_filename(checkpoint_folder):
        checkpoint_filenames = [name for name in os.listdir(checkpoint_folder) if name[-5:]=='.ckpt']
        if len(checkpoint_filenames)==0:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_folder}")

        for name in checkpoint_filenames:
            if '_best' in name:
                checkpoint_filename = name
                break
            checkpoint_filename = name

        return checkpoint_filename


def _get_results_folder(checkpoint_folder: str, checkpoint_filename: str):
    if not os.path.exists(checkpoint_folder):
        raise FileNotFoundError(f"Checkpoint folder not found: {checkpoint_folder}")

    if checkpoint_filename is None:
        checkpoint_filename = _get_checkpoint_filename(checkpoint_folder)

    # create output folder inside the checkpoint folder
    results_folder = '_'.join(checkpoint_filename.split('.')[0].split('_')[:2])
    results_folder += '_results'
    return os.path.join(checkpoint_folder, results_folder)


def compute_positions(net : torch.nn.Module,
                      data_loader : torch.utils.data.DataLoader,
                      device : str = 'cpu',
                      heatmaps_folder : str = None,
                      prefix : str = '',):
    """Compute the position of the ball for all the frames in the dataloader.

    Parameters
    ----------
    net : torch.nn.Module
        the model to evaluate
    data_loader : torch.utils.data.DataLoader
        dataloader of the test set
    device : torch.device, optional
        cpu or cuda, by default cpu
    heatmaps_folder : string, optional
        if provided, saves the heatmaps of each frame

    Returns
    -------
    true_positions : np.ndarray
        list of ground truth positions in (y, x) pixel coordinates
    predicted_positions : np.ndarray
        list of predicted positions in (y, x) pixel coordinates
    min_heatmap_values : np.ndarray
        minimum values of the output activation
    max_heatmap_values : np.ndarray
        maximum values of the output activation
    """
    if heatmaps_folder is not None:
        if not os.path.exists(heatmaps_folder):
            os.makedirs(heatmaps_folder)

    net.eval() # set the model in evaluation mode

    true_positions = []
    predicted_positions = []
    min_heatmap_values = []
    max_heatmap_values = []

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            inputs = data[0].to(device)
            labels = data[1].numpy()

            with torch.autocast(device_type=str(device), dtype=torch.float16):
                outputs = net(inputs).to('cpu').numpy()

            for i, (label, output) in enumerate(zip(labels, outputs)):
                true_positions.append(get_global_maximum(label[-1]))
                predicted_positions.append(get_global_maximum(output[-1]))
                min_heatmap_values.append(output[0].min())
                max_heatmap_values.append(output[0].max())

                if heatmaps_folder is not None:
                    filepath = os.path.join(heatmaps_folder, f"{batch_idx*data_loader.batch_size + i}".zfill(6) + '.png')
                    # normalize heatmap between 0 and 1
                    heatmap = (output[0]-output[0].min()) / (output[0].max()-output[0].min())
                    cv2.imwrite(filepath, np.asarray(heatmap*255, dtype=np.uint8))

            epoch_time = time.time() - start_time
            batch_time = epoch_time/(batch_idx+1)
            remaining_time = batch_time * (len(data_loader)-batch_idx)

            print(prefix + f'{batch_idx+1}/{len(data_loader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, remaining: {remaining_time:.0f}s'.ljust(80), end = '\r')
    print(prefix + f'{batch_idx+1}/{len(data_loader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step'.ljust(80))

    return np.array(true_positions), np.array(predicted_positions), np.array(min_heatmap_values), np.array(max_heatmap_values)


def compute_positions_noh(net : torch.nn.Module,
                          data_loader : torch.utils.data.DataLoader,
                          device : str = 'cpu',
                          prefix : str = '',):
    """Compute the position of the ball for all the frames in the dataloader.

    Parameters
    ----------
    net : torch.nn.Module
        the model to evaluate
    data_loader : torch.utils.data.DataLoader
        dataloader of the test set
    device : torch.device, optional
        cpu or cuda, by default cpu

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
        for batch_idx, data in enumerate(data_loader):
            inputs = data[0].to(device)
            labels = data[1].numpy()

            with torch.autocast(device_type=str(device), dtype=torch.float16):
                outputs = net(inputs).to('cpu').numpy()

            for label, output in zip(labels, outputs):
                true_positions.append(label[::-1])
                predicted_positions.append(output[::-1])

            epoch_time = time.time() - start_time
            batch_time = epoch_time/(batch_idx+1)
            remaining_time = batch_time * (len(data_loader)-batch_idx)

            print(prefix + f'{batch_idx+1}/{len(data_loader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, remaining: {remaining_time:.0f}s'.ljust(80), end = '\r')
    print(prefix + f'{batch_idx+1}/{len(data_loader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step'.ljust(80))

    return np.array(true_positions), np.array(predicted_positions)


def create_output_csv(training_configuration,
                      is_rnn: bool = False,
                      training_phase: str = None,
                      checkpoint_filename: str = None,
                      backup_checkpoint: bool = True,
                      device=None,
                      split: str ='val'):
    """Create the output csv on the test set for the given training configuration

    Parameters
    ----------
    training_configuration : module
        one of the modules in `train_configurations`.
    is_rnn : bool, optional
        set to True if the model is an RNN. The batch size will be set to 1. By default False.
    training_phase : str, optional
        name of the training phase if the checkpoint folder contains subfolders with multiple phases.
        Will raise an error if not provided when training phases are present.
    checkpoint_filename : str, optional
        name of the checkpoint file in the checkpoint folder specified by the training configuration.
        If not provided, the the best checkpoint is used.
    backup_checkpoint : bool, optional
        if True, copy the checkpoint in the output folder. By default True.
    device : torch.device or str, optional
        if not provided, `'cuda'` if available, else `'cpu'`.
    split : str, optional
        `'train'`, `'val'`, `'test'` or `'test_standard'`. By default `'test'`.

    Raises
    ------
    FileNotFoundError
        If the checkpoint folder specified in `training_configuration` is has no checkpoints of if it does not exist
    ValueError
        If `split` is something other than `'train'`, `'val'` or `'test'`.
    """
    config = training_configuration.Config()   # get training configuration

    checkpoint_folder = config._checkpoint_folder
    if training_phase is not None:
        checkpoint_folder = os.path.join(checkpoint_folder, training_phase)

    if checkpoint_filename is None:
        checkpoint_filename = _get_checkpoint_filename(checkpoint_folder)

    results_folder = _get_results_folder(checkpoint_folder, checkpoint_filename)
    if not os.path.exists(results_folder): os.makedirs(results_folder)

    print(f"Saving results in {results_folder}")

    if backup_checkpoint:
        print(f"Copying checkpoint {checkpoint_filename}")
        print(f"From {checkpoint_folder} to {results_folder}")
        shutil.copy2(os.path.join(checkpoint_folder, checkpoint_filename), results_folder)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = config.get_model()
    model.to(device)
    model.load(os.path.join(checkpoint_folder, checkpoint_filename), device=device)
    model.eval()

    if split.lower() in ['train', 'val', 'test']:
        dataset_train, dataset_val, dataset_test = training_configuration.create_datasets()
        if split.lower() == 'train':
            dataset = dataset_train
        elif split.lower() == 'val':
            dataset = dataset_val
        elif split.lower() == 'test':
            dataset = dataset_test
        else:
            raise ValueError("The split must be either 'train', 'val', 'test' or 'test_standard'")
    elif split.lower() == "debug":
        dataset = get_debug_dataset(training_configuration, is_rnn=is_rnn)
    else:
        dataset = get_standard_test_dataset(training_configuration, name=split.lower(), is_rnn=is_rnn)

    batch_size = 1 if is_rnn else config._batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size)

    if hasattr(dataset, "datasets"):
        # depending if dataset is a concatenation of datasets or not
        frames = [frame_num for dataset in dataset.datasets for frame_num in dataset._label_df['num'].values]
        dataset_ids = [i for i, dataset in enumerate(dataset.datasets) for _ in range(len(dataset))]
        dataset_info = dataset.get_info()[0]
    else:
        frames = [frame_num for frame_num in dataset._label_df['num'].values]
        dataset_ids = [0 for _ in range(len(dataset))]
        dataset_info = dataset.get_info()

    frame_offset = 0 if is_rnn else config._sequence_length-1
    output_dict = {'dataset_id': dataset_ids, 'frame_num': [f+frame_offset for f in frames]}

    print("\nComputing results:")
    if dataset_info['output_heatmap']:
        true_positions, predicted_positions, min_values, max_values = compute_positions(
            model,
            data_loader,
            device=device,
            heatmaps_folder=os.path.join(results_folder, f"heatmaps_{split.lower()}"))

        image_size = dataset_info['image_size']
        output_dict['min_values'] = min_values
        output_dict['max_values'] = max_values
        output_dict['x_true'] = true_positions[:,0]/image_size[1]
        output_dict['y_true'] = true_positions[:,1]/image_size[0]
        output_dict['x_pred'] = predicted_positions[:,0]/image_size[1]
        output_dict['y_pred'] = predicted_positions[:,1]/image_size[0]

    else:
        true_positions, predicted_positions = compute_positions_noh(
            model,
            data_loader,
            device=device)
        output_dict['x_true'] = true_positions[:,0]
        output_dict['y_true'] = true_positions[:,1]
        output_dict['x_pred'] = predicted_positions[:,0]
        output_dict['y_pred'] = predicted_positions[:,1]

    pd.DataFrame(output_dict).to_csv(os.path.join(results_folder, f'output_{split.lower()}.csv'), index=False)
    print("done")


def frame_generator(filename, start_frame=None, stop_frame=None, verbose=True):
    """Generator that yields frames from a video.

    Parameters
    ----------
    filename : string
        name of the video file.
    start_frame : int, optional
        starting frame from which to read the video, by default 0
    stop_frame : int, optional
        final frame from which to read the video, by default the final frame
    verbose : bool, optional
        by default True

    Yields
    ------
    array
        the current video frame. The channel order is RGB.

    Raises
    ------
    FileNotFoundError
        if the video file does not exist
    ValueError
        if start_frame >= stop_frame
    """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise FileNotFoundError(f'Video file {filename} not found!')

    if start_frame is None:
        start_frame = 0

    if stop_frame is None:
        stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame >= stop_frame:
        raise ValueError("the starting frame must be smaller than the stopping frame.")

    current_frame = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    ret, frame = cap.read()
    for i in range(start_frame, stop_frame):
        if verbose: print(f"writing frame {i-start_frame+1} of {stop_frame-start_frame}".ljust(80), end = '\r')
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ret, frame = cap.read()
        if not ret:
            if verbose: print("Finished prematurely".ljust(80))
            break
    if verbose: print(f"writing frame {i-start_frame+1} of {stop_frame-start_frame}".ljust(80))
    if verbose: print("Finished frames.")
    cap.release()


def annotate_frame(frame, predicted_positions, true_position=None, max_heatmap_value=None):
    """Put dots on the predicted (red) and true (green) ball positions.

    Parameters
    ----------
    frame : array
        frame to annotate
    predicted_positions : tuple of float
        predicted ball position in pixel coordinates. The coordinate order is `(x, y)`
        can be a list of tuples for multiple detections
    true_position : tuple of float, optional
        true ball position in pixel coordinates. The coordinate order is `(x, y)`
    max_heatmap_value : float or list of float, optional
        the value of the heatmap maximum.
        Can be a list for each detected position.
        If a single value is provided, will be annotated in the upper right corner.
        If a list is provided, will be annotated above each detection.

    Returns
    -------
    annotated_frame : array
        it has the same shape as `frame`.
    """
    annotated_frame = frame.copy()

    if true_position is not None:
        annotated_frame = cv2.circle(annotated_frame,
                                     center=true_position,
                                     radius=5,
                                     color=(0, 255, 0),
                                     thickness=cv2.FILLED)

    if len(np.shape(predicted_positions)) == 1:
        predicted_positions = (predicted_positions, )

    for predicted_position in predicted_positions:
        annotated_frame = cv2.circle(annotated_frame,
                                     center=predicted_position,
                                     radius=5,
                                     color=(255, 0, 0),
                                     thickness=cv2.FILLED)

    annotated_frame = cv2.addWeighted(annotated_frame, 0.6, frame, 0.4, 0)

    if max_heatmap_value is not None:
        if not hasattr(max_heatmap_value, '__iter__'):
            annotated_frame = cv2.putText(annotated_frame,
                                          text=f"{max_heatmap_value:.2g}",
                                          org=(int(0.85*frame.shape[1]), int(0.15*frame.shape[0])),
                                          fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                          fontScale=1,
                                          color=(255, 255, 255))
        else:
            for pos, value in zip(predicted_positions, max_heatmap_value):
                annotated_frame = cv2.putText(annotated_frame,
                                              text=f"{value:.2g}",
                                              org=(pos[0], pos[1] + 20),
                                              fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                              fontScale=0.6,
                                              color=(255, 255, 255))

    return annotated_frame


def create_video(filename_src : str,
                 filename_dst : str,
                 position_df : pd.DataFrame,
                 filename_json : str = None,
                 show_ground_truth : bool = False,
                 heatmaps_folder : str = None,
                 detection_threshold : float = None,
                 local_maxima_smoothing : float = None,
                 start_frame : int = None,
                 stop_frame : int = None,
                 fps : int = None,
                 frame_offset : int = 0):
    """Save the video with the annotated ball position.
    And produce a json with the local maxima for each frame.

    Parameters
    ----------
    filename_src : string
        name of the source video file.
    filename_dst : string
        name of the destination video file.
    position_df : pd.DataFrame
        dataframe with the position data.
        Must contain the following columns:
         - `frame_num`
         - `x_pred`
         - `y_pred`
         - `x_true` (only if `show_ground_truth` is True)
         - `y_true` (only if `show_ground_truth` is True)
    filename_json : str, optional
        name of the json file onto which to save the local maxima.
    show_ground_truth : bool, optional
        by default False
    heatmaps_folder : string, optional
        if provided, heatmaps will be taken from here
    detection_threshold : float, optional
        if provided, annotates the local maxima of the heatmap abore this threshold.
        Otherwise, the global maximum of the heatmap will be used (which is in output.csv).
        Only used if heatmaps_folder is True
    local_maxima_smoothing : float, optional
        if provided, applies gaussian smoothing to the heatmap before computing the local maxima.
        Only used if detection_threshold is provided
    start_frame : int, optional
        if provided, starts the rendering after this many frames.
        By default the first annotated frame in `position_df`.
    stop_frame : int, optional
        if provided, stops the rendering after this many frames.
        The count starts from frame 0 of the original video, and it
        needs to be greater than `start_frame`.
        By default the last annotated frame in `position_df`.
    fps : int, optional
        frames per second of the video, by default the one of the source video file.
    frame_offset : int, optional
        offset frame and heatmap. by default 0
    """

    # get frame range
    if start_frame is None:
        start_frame = position_df['frame_num'].min()
    if stop_frame is None:
        stop_frame = position_df['frame_num'].max() - start_frame

    # read the source video to get fps and resolution
    # and set the resolution of the output video as the one of the input video
    cap = cv2.VideoCapture(filename_src)

    ret, frame_src = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Failed to video in {filename_src}")

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # create the VideoWriter object
    h, w = frame_src.shape[0], frame_src.shape[1]
    out = cv2.VideoWriter(filename=filename_dst,
                          fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                          fps=fps,
                          frameSize=(w, h))

    local_maxima = []

    # loop through the frames of the source video
    for i, frame in enumerate(frame_generator(filename_src, start_frame, start_frame+stop_frame)):
        frame_index = position_df.loc[position_df['frame_num']==i+start_frame + frame_offset].index
        if len(frame_index) != 1:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            continue

        # naive ball position taken from output csv
        x_pred = int(position_df['x_pred'][frame_index]*frame.shape[1])
        y_pred = int(position_df['y_pred'][frame_index]*frame.shape[0])
        predicted_positions = (x_pred, y_pred)

        if show_ground_truth:
            x_true = int(position_df['x_true'][frame_index]*frame.shape[1])
            y_true = int(position_df['y_true'][frame_index]*frame.shape[0])
            true_position = (x_true, y_true)
        else:
            true_position = None

        max_heatmap_value = None
        # add heatmap visualization
        if heatmaps_folder is not None:
            max_heatmap_value = position_df['max_values'][frame_index].values[0]
            min_heatmap_value = position_df['min_values'][frame_index].values[0]
            heatmap = cv2.imread(os.path.join(heatmaps_folder, f"{frame_index.values[0]}".zfill(6)+'.png'))

            if detection_threshold is not None:
                step = 2 # sample the heatmap every 2 steps to improve performance
                normalized_heatmap = heatmap[::step, ::step, 0]/255

                true_heatmap = (normalized_heatmap-min_heatmap_value) * (max_heatmap_value-min_heatmap_value)
                predicted_positions = get_local_maxima(true_heatmap.T, threshold=detection_threshold, sigma=local_maxima_smoothing/step)

                max_heatmap_value = [true_heatmap[pos[1], pos[0]] for pos in predicted_positions]

                predicted_positions[:,0] = step * predicted_positions[:,0] * h/heatmap.shape[0]
                predicted_positions[:,1] = step * predicted_positions[:,1] * w/heatmap.shape[1]

            heatmap = cv2.resize(heatmap, (w, h))
            frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

        frame = annotate_frame(frame, predicted_positions, true_position, max_heatmap_value)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        d = {'frame': i+start_frame}
        if type(max_heatmap_value) is list:
            d['local_maxima'] = [{'value': v,
                                  'x': int(p[1]),
                                  'y': int(p[0])}
                                  for v, p in sorted(zip(max_heatmap_value, predicted_positions), key=lambda x: x[0], reverse=True)]
        else:
            d['local_maxima'] = [{'value': max_heatmap_value,
                                 'x': predicted_positions[0],
                                 'y': predicted_positions[1]}]
        local_maxima.append(d)

    out.release()

    if filename_json is not None:
        with open(filename_json, 'w') as f:
            json.dump(local_maxima, f, indent=2)
        print(f'Local maxima saved in {filename_json}')

    print('done.')


def save_labeled_video(training_configuration,
                       training_phase: str = None,
                       checkpoint_filename: str = None,
                       split: str ='val',
                       dataset_id: Union[int, list] = None,
                       show_ground_truth: bool = True,
                       detection_threshold: float = 0.1,
                       local_maxima_smoothing : float = 5,
                       frame_offset : int = 0):
    """Save the video with the annotated ball position (as a red dot). If `show_ground_truth` is True,
    the ground truth is shown as a green dot.

    Parameters
    ----------
    training_configuration : module
        one of the modules in `train_configurations`
    training_phase : str, optional
        name of the training phase if the checkpoint folder contains subfolders with multiple phases.
        Will raise an error if not provided when training phases are present.
    checkpoint_filename : str, optional
        name of the checkpoint file in the checkpoint folder specified by the training configuration.
        If not provided, the the best checkpoint is used.
    split : str, optional
        `'train'`, `'val'` or `'test'`. By default `'val'`
    dataset_id : int or list of int, optional
        Id of the dataset used in the selected split. By default 0
    show_ground_truth : bool, optional
        Show ground truth on the video. By default True
    detection_threshold : float, optional
        if provided, annotates the local maxima of the heatmap above this threshold.
        Otherwise the position in output.csv is used.
        Used only if the network outputs heatmaps. By default 0.1
    local_maxima_smoothing : float, optional
        if provided, applies gaussian smoothing to the heatmap before computing the local maxima.
        Only used if detection_threshold is provided. By default 5
    frame_offset : int, optional
        offset frame and heatmap. By default 0. It must be set to sequence_length-1 for videos of non-rnn models.

    Raises
    ------
    FileNotFoundError
        If the checkpoint folder specified in `training_configuration` is has no checkpoints of if it does not exist
    FileNotFoundError
        If the specified checkpoint does not have a results folder or if it is empty
    ValueError
        If `split` is something other than `'train'`, `'val'` or `'test'`
    """
    config = training_configuration.Config()   # get training configuration

    checkpoint_folder = config._checkpoint_folder
    if training_phase is not None:
        checkpoint_folder = os.path.join(checkpoint_folder, training_phase)

    results_folder = _get_results_folder(checkpoint_folder, checkpoint_filename)
    if not os.path.exists(results_folder):
        raise FileNotFoundError("The results for the specified checkpoint do not exist. Run `create_output_csv` first.")

    position_df_filename = os.path.join(results_folder, f'output_{split.lower()}.csv')
    position_df = pd.read_csv(position_df_filename)

    heatmaps_folder = os.path.join(results_folder, f'heatmaps_{split.lower()}')
    heatmaps_folder = heatmaps_folder if os.path.exists(heatmaps_folder) else None

    if split.lower() in ['train', 'val', 'test']:
        with open(os.path.join(config._checkpoint_folder, f"dataset_{split.lower()}_info.json"), 'r') as f:
            dataset_info = json.load(f)

        if dataset_id is None:
            dataset_id = [i for i in range(len(dataset_info))]
        elif type(dataset_id) is int:
            dataset_id = [dataset_id]
    else:
        dataset_info = [{'root': f'../datasets/{split.lower()}'}]
        dataset_id = [0]
        show_ground_truth = False

    print(f"Using data from {position_df_filename}\n")
    for i in dataset_id:
        src = os.path.join(dataset_info[i]['root'], 'video.mp4')
        dst = os.path.join(results_folder, f'video_{split.lower()}_{i}.mp4')
        filename_json = os.path.join(results_folder, f'video_{split.lower()}_{i}.json')

        print(f"Video {i+1} of {len(dataset_id)}")
        print(f"src file: {src}")
        print(f"dst file: {dst}")

        create_video(filename_src=src,
                     filename_dst=dst,
                     filename_json=filename_json,
                     position_df=position_df.loc[position_df['dataset_id']==i],
                     show_ground_truth=show_ground_truth,
                     heatmaps_folder=heatmaps_folder,
                     detection_threshold=detection_threshold,
                     local_maxima_smoothing=local_maxima_smoothing,
                     frame_offset=frame_offset)
        print()
