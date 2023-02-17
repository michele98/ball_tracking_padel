import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Callable

import torch
from torch.utils.data import Dataset


#TODO: add support for other video formats
#TODO: add option for removing non annotated frames if they come up in a frame sequence
class VideoDataset(Dataset):
    def __init__(self,
                 root: str,
                 output_heatmap: bool = True,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 concatenate_sequence: bool = True,
                 sequence_length: int = 3,
                 overlap_sequences: bool = True,
                 image_size: Tuple[int, int] = None,
                 sigma: float = 5,
                 drop_duplicate_frames: bool = True,
                 duplicate_equality_threshold: float = 0.97):
        """Dataset that starts from a video and csv file with the ball position in each frame.
        The coordinates in the csv file are pixel coordinates, normalized between 0 and 1.
        Outputs a sequence of consecutive frames and the cooresponding ball pixel coordinates or heatmaps.

        The first frame of the sequence always has the labeled coordinates.
        If the next frames in the sequence are not labelled:
         - if `output_heatmap` is set to True a heatmap containing only `0` is returned
         - if `output_heatmap` is set to False `(None, None)` is returned


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
        target_transform : Callable, optional
            transformation to apply to the output
        concatenate_sequence : bool, optional
            Concatenate frames and heatmaps along the channel dimension. By default True
        sequence_length : int, optional
            length of the video frame sequence. By default 3
        overlap_sequences : bool, optional
            if set to True, the starting frame is picked randomly, so different sequences might overlap.
            if set to False, starting frames are sampled only in multiples of `sequence_length`,
            so each frame sequence is independent from the others. By default True
        image_size : Tuple[int, int], optional
            resize the frames to `(height, width)`. If not provided, the original size is kept
        sigma : float, optional
            the width in pixels of the gaussian centered on the ball. Used only if `output_heatmap` is set to True.
            By default 5
        drop_duplicate_frames : bool, optional
            Choose wheter to drop duplicate frames from the frame sequence.
            Attention, this might make the `__len__()` method unreliable, since the number of duplicate frames is not known beforehand.
            By default True
        duplicate_equality_threshold : float, optional
            Frames are considered equal if the fraction of pixels with the same value between 2 frames is larger than this.
            By default 0.97
        """
        self.root = root
        self.output_heatmap = output_heatmap
        self.transform = transform
        self.target_transform = target_transform
        self.concatenate_sequence = concatenate_sequence
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
        self.drop_duplicate_frames = drop_duplicate_frames
        self.duplicate_equality_threshold = duplicate_equality_threshold

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

    def _equal_frames(self, frame_1, frame_2):
        """Check if frames are equal. Returns False if either one is None."""
        if frame_1 is None or frame_2 is None:
            return False

        absolute_difference = np.abs(frame_1 - frame_2)
        num_equal = np.count_nonzero(absolute_difference==0)
        num_tot = np.prod(absolute_difference.shape)

        return num_equal/num_tot > self.duplicate_equality_threshold

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
        size = 5*self.sigma

        heatmap = np.zeros(self.image_size, dtype=np.float32)
        heatmap = np.pad(heatmap, size)

        y, x = self._get_coordinates(frame_number)

        if x is not None and y is not None:
            x = int(x)
            y = int(y)

            x_grid, y_grid = np.mgrid[-size:size + 1, -size:size + 1]
            g = np.exp(-(x_grid**2 + y_grid**2) / float(2 * self.sigma**2))
            g /= 2*np.pi*self.sigma**2

            heatmap[y:y + (size*2) + 1, x:x + (size*2) + 1] = g
            heatmap = heatmap[size:-size, size:-size]

        if self.target_transform is not None:
            heatmap = self.target_transform(heatmap)

        return heatmap

    def __len__(self):
        if self.overlap_sequences:
            return len(self.label_df)
        return len(self.label_df)//self.sequence_length

    def __getitem__(self, idx):
        if idx<0:
            idx += len(self.label_df)

        if self.overlap_sequences:
            item_idx = idx
        else:
            item_idx = idx*self.sequence_length

        starting_frame_number = self.label_df['num'][item_idx]

        frames = []
        labels = []

        i = 0 # index for counting the frames in the sequence
        j = 0 # index for keeping track of duplicate frames
        last_used_frame = None
        #for i in range(self.sequence_length):
        while i < self.sequence_length:
            frame_number = starting_frame_number+i+j
            frame = self._read_frame(frame_number)
            if self._equal_frames(last_used_frame, frame) and self.drop_duplicate_frames:
                j+=1
                continue
            last_used_frame = frame
            i+=1

            frames.append(frame)
            labels.append(self._generate_heatmap(frame_number) if self.output_heatmap else self._get_coordinates(frame_number))

        if self.concatenate_sequence and type(labels[0]) is torch.Tensor and type(frames[0]) is torch.Tensor:
            frames = torch.cat(frames, dim=0)
            labels = torch.cat(labels, dim=0)

        return frames, labels
