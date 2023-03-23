import os
import cv2
import json
import numpy as np
import pandas as pd
from typing import Tuple, Callable

import torch
from torch.utils.data import Dataset, ConcatDataset


#TODO: add support for other video formats
#TODO: add option for removing non annotated frames if they come up in a frame sequence
class VideoDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str = None,
                 output_heatmap: bool = True,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 concatenate_sequence: bool = True,
                 sequence_length: int = 3,
                 overlap_sequences: bool = True,
                 image_size: Tuple[int, int] = None,
                 sigma: float = 5,
                 heatmap_mode = 'normalized',
                 drop_duplicate_frames: bool = True,
                 duplicate_equality_threshold: float = 0.97,
                 one_output_frame: bool = True,
                 grayscale: bool = False,
                 preload_in_memory: bool = True):
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
             - labels[_split].csv: must contain the following attributes:
                 - `num` (int): frame number;
                 - `x` (float): normalized horizontal coordinate from the left border;
                 - `y` (float): normalized vertical coordinate from the upper border;
                 - `visibility` (int): 0 (occluded), 1 (visible), 2 (motion blurred), 3 (unknown).

        split : str, optional
            `'train'`, `'val'` or `'test'`.
            If provided, the file named `labels_[split].csv` will be used.
            Otherwise, the file named `labels.csv` will be used.
        output_heatmap : bool, optional
            if set to True, outputs a heatmap with the probability of finding the ball in a specific pixel.
            Otherwise outputs only the normalized ball pixel coordinates as (y, x). By default True
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
        heatmap_mode : str, optional
            Can be either:
             - `'normalized'`: the heatmap is a Gaussian normal distribution, i.e. the integral over the whole heatmap is 1;
             - `'image'`: the maximum value is set to 255, and the data type is uint8.
            By default `'normalized'`
        drop_duplicate_frames : bool, optional
            Choose wheter to drop duplicate frames from the frame sequence.
            Attention, this might make the `__len__()` method unreliable, since the number of duplicate frames is not known beforehand.
            By default True
        duplicate_equality_threshold : float, optional
            Frames are considered equal if the fraction of pixels with the same value between 2 frames is larger than this.
            By default 0.97
        one_output_frame : bool, optional
            if set to True, outputs a single heatmap only for the last frame.
            If set to False, outputs a heatmap for each input frame.
            By default True
        grayscale : bool, optional
            if true, the frames are loaded in grayscale. By default False
            NOTE: FOR NOW IT IS DISABLED
        preload_in_memory : bool, optional
            Preloads the frames to RAM. It improves loading times in training, but could lead to high memory usage. By default True
            NOTE: FOR NOW IT IS DISABLED
        """
        self.root = root
        self.output_heatmap = output_heatmap
        self.transform = transform
        self.target_transform = target_transform
        self.concatenate_sequence = concatenate_sequence
        self.split = split
        if split is None:
            self._label_df = pd.read_csv(os.path.join(root, f"labels.csv"))
        else:
            self._label_df = pd.read_csv(os.path.join(root, f"labels_{split}.csv"))
        self.sequence_length = sequence_length
        self.overlap_sequences = overlap_sequences
        self._cap = cv2.VideoCapture(os.path.join(root, "video.mp4"))
        if image_size is None:
            _, frame = self._cap.read()
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.image_size = frame.shape[:2]
        else:
            self.image_size = image_size
        self.sigma = sigma
        if heatmap_mode.lower() != 'normalized' and heatmap_mode.lower() != 'image':
            raise ValueError("heatmap_mode must either be 'normalized' or 'image'")
        self.heatmap_mode = heatmap_mode.lower()
        self.drop_duplicate_frames = drop_duplicate_frames
        self.duplicate_equality_threshold = duplicate_equality_threshold
        self.one_output_frame = one_output_frame

        # TODO: implement grayscale option
        self.grayscale = grayscale
        self.preload_in_memory = preload_in_memory
        self.last_read_frame = -1

        self._frames_to_preload = self._get_frames_to_preload()
        self._preload_LUT = {f: i for i, f in enumerate(self._frames_to_preload)}

        # TODO: Make memory pre-loading work properly
        if preload_in_memory:
            self._preload_frames()
            self._cap.release()

    def _get_frames_to_preload(self):
        frame_numbers = self._label_df['num'].values
        old_frame = frame_numbers[0]
        frames_to_preload = [old_frame]

        margin=1 # margin for duplicate frames

        # pad gaps in indices of labelled frames
        for frame_number in frame_numbers[1:]:
            if frame_number != old_frame+1:
                for i in range(self.sequence_length+margin-1):
                    to_add = old_frame+i+1
                    if to_add == frame_number:
                        break
                    frames_to_preload.append(to_add)
            frames_to_preload.append(frame_number)
            old_frame = frame_number

        for i in range(self.sequence_length+margin-1):
            frames_to_preload.append(old_frame+i+1)

        return frames_to_preload

    def _get_normalized_coordinates(self, frame_number):
        idx = self._label_df.loc[self._label_df['num']==frame_number].index
        if len(idx)==0:
            return None
        x = self._label_df['x'].iloc[idx].values[0]
        y = self._label_df['y'].iloc[idx].values[0]
        return y, x

    def _get_coordinates(self, frame_number):
            normalized_coordinates = self._get_normalized_coordinates(frame_number)
            if normalized_coordinates is not None:
                return [normalized_coordinates[i]*self.image_size[i] for i in range(2)]
            else:
                return None, None

    # TODO: utilize visibility info
    def _generate_heatmap(self, frame_number):
        size = 5*self.sigma

        if self.heatmap_mode == 'normalized':
            dtype=np.float32
        else:
            dtype=np.uint8
        heatmap = np.zeros(self.image_size, dtype=dtype)

        y, x = self._get_coordinates(frame_number)

        if x is not None and y is not None:
            x = int(x)
            y = int(y)

            x_grid, y_grid = np.mgrid[-size:size + 1, -size:size + 1]
            g = np.exp(-(x_grid**2 + y_grid**2) / float(2 * self.sigma**2))

            if self.heatmap_mode == 'normalized':
                g /= 2*np.pi*self.sigma**2
            else:
                g = np.asarray(g*255/np.max(g), dtype=np.uint8)

            heatmap = np.pad(heatmap, size)
            heatmap[y:y + (size*2) + 1, x:x + (size*2) + 1] = g
            heatmap = heatmap[size:-size, size:-size]

        return heatmap

    def _equal_frames(self, frame_1, frame_2):
        """Check if frames are equal. Returns False if either one is None."""
        if frame_1 is None or frame_2 is None:
            return False

        absolute_difference = np.abs(frame_1 - frame_2)
        num_equal = np.count_nonzero(absolute_difference==0)
        num_tot = np.prod(absolute_difference.shape)

        return num_equal/num_tot >= self.duplicate_equality_threshold

    def _read_frame(self, frame_number):
            if self.last_read_frame+1 != frame_number:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.last_read_frame = frame_number

            ret, frame = self._cap.read()
            if ret:
                return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.image_size[::-1])
            else:
                print(f"Attention! Failed to read frame {frame_number}")
                return None

    def _preload_frames(self):
        print("Loading frames:")

        # num_channels = 1 if self.grayscale else 3
        num_channels = 3
        try:
            self.frames = np.zeros((len(self._frames_to_preload), *self.image_size, num_channels), dtype=np.uint8)
        except MemoryError as e:
            print(e)
            print("Frames will be read from disk")
            self.preload_in_memory = False
            return

        for i, frame_idx in enumerate(self._frames_to_preload):
            if i%10==0:
                print(f"Loaded {i} of {len(self._frames_to_preload)}", end='\r')
            self.frames[i] = self._read_frame(frame_idx)
        print("Done".ljust(50))

    def _get_frame(self, frame_number):
        if self.preload_in_memory:
            return self.frames[self._preload_LUT[frame_number]]
        return self._read_frame(frame_number)

    def __len__(self):
        if self.overlap_sequences:
            return len(self._label_df)
        return len(self._label_df)//self.sequence_length

    def __getitem__(self, idx):
        if idx<0:
            idx += self.__len__()

        if self.overlap_sequences:
            item_idx = idx
        else:
            item_idx = idx*self.sequence_length

        starting_frame_number = self._label_df['num'][item_idx]

        frames = []
        if not self.one_output_frame: labels = []

        i = 0 # index for counting the frames in the sequence
        j = 0 # index for keeping track of duplicate frames
        last_used_frame = None

        while i < self.sequence_length:
            frame_number = starting_frame_number+i+j
            frame = self._get_frame(frame_number)
            if self.drop_duplicate_frames and self._equal_frames(last_used_frame, frame):
                j+=1
                continue
            last_used_frame = frame
            i+=1
            if self.transform is not None:
                frame = self.transform(frame)
            frames.append(frame)
            if not self.one_output_frame:
                label = self._generate_heatmap(frame_number) if self.output_heatmap else self._get_normalized_coordinates(frame_number)
                if self.target_transform is not None:
                    label = self.target_transform(label)
                labels.append(label)

        if self.one_output_frame:
            labels = self._generate_heatmap(frame_number) if self.output_heatmap else self._get_normalized_coordinates(frame_number)
            if self.target_transform is not None:
                labels = self.target_transform(labels)

        if type(frames[0]) is torch.Tensor:
            if self.concatenate_sequence:
                frames = torch.cat(frames, dim=0)
                if not self.one_output_frame:
                    labels = torch.cat(labels, dim=0)

        return frames, labels

    def get_info(self):
        """Has all the info that was used to cheate the current instance of the `VideoDataset`.

        Returns
        -------
        dict
            It contains the values of the arguments passed to the constructor.
        """
        output_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        output_dict['len'] = self.__len__()
        return output_dict

    def save_info(self, output_filename):
        with open(output_filename, 'w') as f:
            json.dump(self.get_info(), f, default=str, indent=2)


class MyConcatDataset(ConcatDataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    In addition to `ConcatDataset`, it has the `get_info` method, which returns a list of information dictionaries
    from the `get_info` method of `VideoDataset`.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    def get_info(self):
        """
        Returns
        -------
        list of dict
            list of info dictionaries for each `VideoDataset`.
        """
        return [dataset.get_info() for dataset in self.datasets]

    def save_info(self, output_filename):
        with open(output_filename, 'w') as f:
            json.dump(self.get_info(), f, default=str, indent=2)
