import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Callable
from torch.utils.data import Dataset


#TODO: add support for other video formats
#TODO: add option for removing non annotated frames if they come up in a frame sequence
class VideoDataset(Dataset):
    def __init__(self, root : str,
                 output_heatmap=True,
                 transform : Callable = None,
                 sequence_length=3,
                 overlap_sequences=True,
                 image_size : Tuple[int, int] = None,
                 sigma : float = 5):
        """Dataset that starts from a video and csv file with the ball position in each frame.
        The coordinates in the csv file are pixel coordinates, normalized between 0 and 1.
        Outputs a frame sequence and the cooresponding ball pixel coordinates or heatmaps.

        Parameters
        ----------
        root : str
            directory containing the video and csv with the ball position at each frame.
            it must contain the following files:
             - video.mp4: video file
             - labels.csv: must contain the following attributes:
                 - `num` (int): frame number;
                 - `x` (float): normalized horizontal coordinate from the left border;
                 - `y` (float): normalized vertical coordinate from the upper border;
                 - `visibility` (int): 0 (occluded), 1 (visible), 2 (motion blurred).

        output_heatmap : bool, optional
            if set to True, outputs a heatmap with the probability of finding the ball in a specific pixel.
            Otherwise outputs only the ball coordinates. By default True
        transform : Callable, optional
            transformation to apply to the video frames
        sequence_length : int, optional
            length of the video frame sequence. By default 3
        overlap_sequences : bool, optional
            if set to True, the starting frame is picked randomly, so different sequences might overlap.
            if set to False, starting frames are sampled only in multiples of `sequence_length`,
            so each frame sequence is independent from the others. By default True
        image_size : Tuple[int, int], optional
            resize the frames to `(height, width)`. If not provided, the original size is kept
        sigma : float, optional
            the width in pixels of the gaussian centered on the ball. Used only if output_heatmap` is set to True.
            By default 5
        """
        self.root = root
        self.output_heatmap = output_heatmap
        self.transform = transform
        self.label_df = pd.read_csv(os.path.join(root, "labels.csv"))
        self.sequence_length = sequence_length
        self.overlap_sequences = overlap_sequences
        self.cap = cv2.VideoCapture(os.path.join(root, "video.mp4"))
        if image_size is None:
            _, frame = self.cap.read()
            self.image_size = frame.shape[:2]
        else:
            self.image_size = image_size
        self.sigma = sigma

    def _read_frame(self, frame_number):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                resized_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.image_size[::-1])
                if self.transform is not None:
                    resized_frame = self.transform(resized_frame)
                return resized_frame
            else:
                print(f"Attention! Failed to read {frame_number}")
                return None #TODO: make it handle this None

    def _get_normalized_coordinates(self, frame_number):
        idx = self.label_df.loc[self.label_df['num']==frame_number].index
        if len(idx)==0:
            return None
        x = self.label_df['x'].iloc[idx].values[0]
        y = self.label_df['y'].iloc[idx].values[0]
        return y, x

    def _get_coordinates(self, frame_number):
            normalized_coordinates = self._get_normalized_coordinates(frame_number)
            if normalized_coordinates is not None:
                return [normalized_coordinates[i]*self.image_size[i] for i in range(2)]
            else:
                return None, None

    # TODO: utilize visibility info
    # TODO: check padding info
    def _generate_heatmap(self, frame_number):
        y, x = self._get_coordinates(frame_number)
        x = int(x)
        y = int(y)
        size = 5*self.sigma

        x_grid, y_grid = np.mgrid[-size:size + 1, -size:size + 1]
        g = np.exp(-(x_grid**2 + y_grid**2) / float(2 * self.sigma**2))
        g /= 2*np.pi*self.sigma**2

        heatmap = np.zeros(self.image_size)
        heatmap = np.pad(heatmap, size)

        heatmap[y:y + (size*2) + 1, x:x + (size*2) + 1] = g
        heatmap = heatmap[size:-size, size:-size]
        return heatmap

    def __len__(self):
        if self.overlap_sequences:
            return len(self.label_df)
        return len(self.label_df)//self.sequence_length

    def __getitem__(self, idx):
        if self.overlap_sequences:
            item_idx = idx
        else:
            item_idx = idx*self.sequence_length

        starting_frame = self.label_df['num'][item_idx]

        frames = []
        labels = []
        for i in range(self.sequence_length):
            frames.append(self._read_frame(starting_frame+i))
            labels.append(self._generate_heatmap(starting_frame+i) if self.output_heatmap else self._get_coordinates(starting_frame+i))

        return frames, labels
