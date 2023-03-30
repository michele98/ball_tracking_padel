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
            Attention, when sampling, duplicate frames are treated as separate frames.
            By default True
        duplicate_equality_threshold : float, optional
            Frames are considered equal if the fraction of pixels with the same value between 2 frames is larger than this.
            By default 0.97
        one_output_frame : bool, optional
            if set to True, outputs a single heatmap only for the last frame.
            If set to False, outputs a heatmap for each input frame.
            By default True
        grayscale : bool, optional
            if True, the frames are loaded in grayscale. By default False
        preload_in_memory : bool, optional
            preload the frames to RAM. It improves loading times in training, but could lead to high memory usage. By default True
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
        self.grayscale = grayscale

        self._preload_in_memory = preload_in_memory
        self._preload_in_memory_hm = output_heatmap and preload_in_memory
        self._last_read_frame = -1

        if preload_in_memory:
            self._frames_to_preload, self._padding_frames = self._get_frames_to_preload()
            self._preload_in_memory = self._allocate_frames()
            if self.output_heatmap:
                self._preload_in_memory_hm = self._allocate_heatmaps()

            if self._preload_in_memory:
                self._preload()
            self._cap.release()

    def _get_frames_to_preload(self):
        frame_numbers = sorted(self._label_df['num'].values)
        old_frame_number = frame_numbers[0]
        frames_to_preload = [old_frame_number]
        padding_frames = [False] # mask that says if the frame is labelled or not

        # pad gaps in indices of labelled frames
        for frame_number in frame_numbers[1:]:
            # if there is a gap, pad it
            if frame_number > old_frame_number+1:
                for i in range(self.sequence_length-1):
                    pad_frame_number = old_frame_number+i+1
                    if pad_frame_number == frame_number:
                        break
                    frames_to_preload.append(pad_frame_number)
                    padding_frames.append(True)
            frames_to_preload.append(frame_number)
            padding_frames.append(False)
            old_frame_number = frame_number

        # pad last frame
        for i in range(self.sequence_length-1):
            frames_to_preload.append(old_frame_number+i+1)
            padding_frames.append(True)

        return frames_to_preload, padding_frames

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
            if self._last_read_frame+1 != frame_number:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self._last_read_frame = frame_number

            ret, frame = self._cap.read()
            flag = cv2.COLOR_BGR2GRAY if self.grayscale else cv2.COLOR_BGR2RGB
            if ret:
                return cv2.resize(cv2.cvtColor(frame, flag), self.image_size[::-1])
            else:
                print(f"Attention! Failed to read frame {frame_number}")
                return None

    def _allocate_frames(self):
        sample_frame = self._read_frame(self._label_df['num'][0])
        if self.transform is not None:
            sample_frame = self.transform(sample_frame)
        try:
            channels_per_image = 1 if self.grayscale else 3
            if torch.is_tensor(sample_frame):
                self._frames = torch.zeros((len(self._frames_to_preload), channels_per_image, *self.image_size), dtype=torch.float16)
            else:
                if self.grayscale:
                    self._frames = np.zeros((len(self._frames_to_preload), *self.image_size), dtype=np.uint8)
                else:
                    self._frames = np.zeros((len(self._frames_to_preload), *self.image_size, channels_per_image), dtype=np.uint8)
            return True
        except (MemoryError, RuntimeError) as e:
            print(e)
            print("Frames will be read from disk")
            return False

    def _allocate_heatmaps(self):
        if not self.output_heatmap:
            return False

        sample_heatmap = self._generate_heatmap(self._label_df['num'][0])
        if self.target_transform is not None:
            sample_heatmap = self.target_transform(sample_heatmap)
        try:
            if torch.is_tensor(sample_heatmap):
                self._heatmaps = torch.zeros((len(self._frames_to_preload), *self.image_size), dtype=torch.float16)
            else:
                self._heatmaps = np.zeros((len(self._frames_to_preload), *self.image_size), dtype=np.uint8)
            return True
        except (MemoryError, RuntimeError) as e:
            print(e)
            print("Heatmaps will be generated as needed")
            return False

    def _preload(self):
        print("Loading frames:", end = '\r')
        self._frame_LUT = {}

        j = 0 # index for keeping track of LUT offset due to duplicate frames
        last_used_frame = None

        for i, frame_number in enumerate(self._frames_to_preload):
            frame = self._read_frame(frame_number)
            if self.drop_duplicate_frames:
                # increase frame number of the next padding frames if the next frame is a padding frame and equal to the current frame
                if (
                        i+1 < len(self._frames_to_preload) and
                        self._padding_frames[i+1] and
                        self._equal_frames(frame, self._read_frame(frame_number+1))
                    ):
                    for k in range(1, self.sequence_length):
                        if i+k >= len(self._padding_frames) or not self._padding_frames[i+k]:
                            break
                        self._frames_to_preload[i+k]+=1
                # offset index if frames are equal
                if not self._padding_frames[i] and self._equal_frames(last_used_frame, frame):
                    j += 1
            last_used_frame = frame

            if self.transform is not None:
                frame = self.transform(frame)

            self._frame_LUT[frame_number] = i-j
            self._frames[i-j] = frame
            if self.output_heatmap and self._preload_in_memory_hm:
                heatmap = self._generate_heatmap(frame_number)
                if self.target_transform is not None:
                    heatmap = self.target_transform(heatmap)
                self._heatmaps[i-j] = heatmap
            print(f"Loading frames: {i+1} of {len(self._frames_to_preload)}", end='\r')
        print(f"Loading frames: {i+1} of {len(self._frames_to_preload)}.".ljust(32), "Done")

    def __len__(self):
        if self.overlap_sequences:
            return len(self._label_df)
        return len(self._label_df)//self.sequence_length

    def _get_item_old(self, item_idx):
        starting_frame_number = self._label_df['num'][item_idx]

        frames = []
        if not self.one_output_frame: labels = []

        i = 0 # index for counting the frames in the sequence
        j = 0 # index for keeping track of duplicate frames
        last_used_frame = None

        while i < self.sequence_length:
            frame_number = starting_frame_number+i+j
            frame = self._read_frame(frame_number)
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
                if self.output_heatmap and not self.one_output_frame:
                    labels = torch.cat(labels, dim=0)

        return frames, labels

    def __getitem__(self, idx):
        if idx<0:
            idx += self.__len__()
        if self.overlap_sequences:
            item_idx = idx
        else:
            item_idx = idx*self.sequence_length

        if not self._preload_in_memory:
            return self._get_item_old(item_idx)

        starting_frame_number = self._label_df['num'][item_idx]
        i = self._frame_LUT[starting_frame_number]

        # TODO: refactor this monstruosity
        if self.output_heatmap:
            if self._preload_in_memory_hm:
                if self.one_output_frame:
                    heatmaps = self._heatmaps[i+self.sequence_length-1]
                    if torch.is_tensor(heatmaps):
                        heatmaps = heatmaps.view(1, *heatmaps.shape)
                else:
                    heatmaps = self._heatmaps[i:i+self.sequence_length]
            else:
                heatmaps = self._generate_heatmap(starting_frame_number)
        else:
            if self.one_output_frame:
                heatmaps = self._get_normalized_coordinates(starting_frame_number+self.sequence_length-1)
                if self.target_transform is not None:
                    heatmaps = self.target_transform(heatmaps)
            else:
                heatmaps = [self._get_normalized_coordinates(starting_frame_number+i) for i in range(self.sequence_length)]
                if self.target_transform is not None:
                    for i in range(len(heatmaps)):
                        heatmaps[i] = self.target_transform(heatmaps[i])

        frames = self._frames[i:i+self.sequence_length]
        channels_per_image = 1 if self.grayscale else 3
        if torch.is_tensor(frames):
            frames = frames.view((channels_per_image*self.sequence_length, *self.image_size))
        else:
            if self.grayscale:
                frames = frames.reshape((self.sequence_length, *self.image_size))
            else:
                frames = frames.reshape((self.sequence_length, *self.image_size, channels_per_image))
        return frames, heatmaps

    def get_info(self):
        """Has all the info that was used to cheate the current instance of `VideoDataset`.

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


class VideoDatasetRNN(VideoDataset):
    def __init__(self, *args, **kwargs):
        """Dataset for RNN. The input consists of the previous heatmaps and the current frame.
        The output is the heatmap to predict.
        The following parameters are forced:
         - `output_heatmap=True`
         - `heatmap_mode='image'`
         - `preload_in_memory=True`
         - `drop_duplicate_frames=False`

        Parameters
        ----------
        **kwargs : passed to :class:`VideoDataset`
        """
        super().__init__(*args, output_heatmap=True, heatmap_mode='image', preload_in_memory=True, **kwargs)

    def _get_frames_to_preload(self):
        # here we want to pad the frame before the first labeled one instead of after the last labelled
        frame_numbers = sorted(self._label_df['num'].values)
        old_frame_number = frame_numbers[0]

        # pad first frames
        frames_to_preload = [old_frame_number-i for i in reversed(range(self.sequence_length))]
        padding_frames = [True for _ in range(self.sequence_length-1)] + [False] # mask that says if the frame is labelled or not

        # pad gaps in indices of labelled frames
        for frame_number in frame_numbers[1:]:
            # if there is a gap, pad it
            if frame_number > old_frame_number+1:
                for i in reversed(range(self.sequence_length-1)):
                    pad_frame_number = frame_number-i-1
                    if pad_frame_number <= old_frame_number:
                        continue
                    frames_to_preload.append(pad_frame_number)
                    padding_frames.append(True)
            frames_to_preload.append(frame_number)
            padding_frames.append(False)
            old_frame_number = frame_number

        return frames_to_preload, padding_frames

    def __getitem__(self, idx):
        if idx<0:
            idx += self.__len__()
        if self.overlap_sequences:
            item_idx = idx
        else:
            item_idx = idx*self.sequence_length

        i = self._frame_LUT[self._label_df['num'][item_idx]]

        frame = self._frames[i]
        heatmaps = self._heatmaps[i-self.sequence_length+1:i+1]

        input = (heatmaps[:-1], frame)
        output = heatmaps[-1] if self.one_output_frame else heatmaps

        if torch.is_tensor(frame) and torch.is_tensor(heatmaps):
            if self.one_output_frame:
                output = output.view(1, *self.image_size)
            return torch.concat(input), output

        return input, output


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
