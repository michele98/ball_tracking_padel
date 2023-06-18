import os
import cv2
import json
import numpy as np
import pandas as pd
from detection.testing import _get_results_folder


def get_video_source(training_configuration, split='val_1'):
    config = training_configuration.Config()

    # open dataset info json
    if 'val' in split or 'train' in split:
        with open(os.path.join(config._checkpoint_folder, f"dataset_{split.split('_')[0]}_info.json"), 'r') as f:
            dataset_info = json.load(f)
        dataset_info = dataset_info[int(split.split('_')[1])]
        video_source = os.path.join(dataset_info['root'], 'video.mp4')
    else:
        video_source = os.path.join('../datasets', split, 'video.mp4')

    return video_source


def get_frame(training_configuration, frame_index, split='val_1', size=None):
    video_source = get_video_source(training_configuration, split)

    # print(f'Video source: {video_source}')
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        print('Failed to read frame')
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if size is not None:
        frame = cv2.resize(frame, size)
    return frame


def get_heatmap(training_configuration, frame_index, split='val_1', training_phase=None):
    config = training_configuration.Config()

    checkpoint_folder = config._checkpoint_folder
    if training_phase is not None:
        checkpoint_folder = os.path.join(checkpoint_folder, training_phase)

    results_folder = _get_results_folder(checkpoint_folder, None)

    if 'val' in split or 'train' in split:
        df = pd.read_csv(os.path.join(results_folder, f"output_{split.split('_')[0]}.csv"))
        heatmaps_folder = os.path.join(results_folder, f"heatmaps_{split.split('_')[0]}")
        dataset_id = int(split.split('_')[1])
    else:
        df = pd.read_csv(os.path.join(results_folder, f"output_{split}.csv"))
        heatmaps_folder = os.path.join(results_folder, f"heatmaps_{split}")
        dataset_id = 0

    heatmap_index = df[(df['dataset_id'] == dataset_id) & (df['frame_num'] == frame_index)].index

    if len(heatmap_index) != 1:
        return None

    heatmap = cv2.imread(os.path.join(heatmaps_folder, f"{heatmap_index[0]}".zfill(6)+'.png'), cv2.IMREAD_GRAYSCALE)
    return (heatmap*df['max_values'][heatmap_index[0]]).astype(np.uint8)



def get_candidates_json(training_configuration, training_phase: str = None, split: str = 'val_1'):
    """Get the local maxima json of the given train configuration"""
    config = training_configuration.Config()

    checkpoint_folder = config._checkpoint_folder
    if training_phase is not None:
        checkpoint_folder = os.path.join(checkpoint_folder, training_phase)

    if not 'val' in split and not 'train' in split:
        split += '_0'

    results_folder = _get_results_folder(checkpoint_folder, None)
    candidates_fileame = os.path.join(results_folder, f'video_{split}.json')
    with open(candidates_fileame, 'r') as f:
        d = json.load(f)
    return d


def get_candidates(training_configuration, training_phase: str = None, split: str = 'val_1', max_num_candidates: int = 20):
    """Get the local maxima arrays of the given train configuration.
    It reads the data from the json file generated with the detection video.

    Parameters
    ----------
    training_configuration : module
        train configuration
    training_phase : str, optional
        must be provided for RNN architectures, by default None
    split : str, optional
        `'train_0'` to `'train_4'`, `'val_0'`, `'val_1'`, or other unlabeled splits. By default 'val_1'
    max_num_candidates : int, optional
        maximum number of position candidates per frame. By default 20

    Returns
    -------
    starting_frame : int
        index of the frame of the original video from which the detection starts
    candidates : np.ndarray of int, shape `(num_frames, max_num_candidates, 2)`
        contains the positions of the detection candidates
    n_candidates : np.ndarray of int, shape `(num_frames,)`
        number of candidates for each frame
    values : np.ndarray of float, shape `(num_frames, max_num_candidates)`
        value of the local maximum corresponding to the candidate
    """
    detections = get_candidates_json(training_configuration, training_phase, split)

    starting_frame, end_frame = detections[0]['frame'], detections[-1]['frame']
    num_frames = end_frame - starting_frame + 1

    n_candidates = np.zeros(num_frames, dtype=int) # number of the maxima
    candidates = np.zeros((num_frames, max_num_candidates, 2), dtype=int) - 1 # x-y coordinates of the maxima
    values = np.zeros((num_frames, max_num_candidates), dtype=float) # maximum values of the maxima

    for d in detections:
        i = d['frame'] - starting_frame
        n_candidates[i] = len(d['local_maxima'])
        for j, local_maximum in enumerate(d['local_maxima']):
            if j >= max_num_candidates:
                break
            candidates[i,j,0] = local_maximum['x']
            candidates[i,j,1] = local_maximum['y']
            values[i,j] = local_maximum['value']

    return starting_frame, candidates, n_candidates, values
