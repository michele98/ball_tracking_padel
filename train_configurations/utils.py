from detection.dataset import VideoDataset, VideoDatasetRNN, MyConcatDataset

import torch
from torch.utils.data._utils.collate import default_collate
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import Sampler
from torchvision.transforms import ToTensor

def collate_fn_rnn(batch,
                   total_clear_probability: float,
                   clear_probability: float,
                   ground_truth_probability: list,
                   sequence_length: int):
    """Base collate function to pass to a DataLoader for TrackNetV2RNN.
    When passing to a DataLoader, fix all the parameters other than `batch`

    Parameters
    ----------
    batch : batch of inputs and labels
    clear_probability : float
        probability that the previous heatmaps are cleared.
    ground_truth_probability : float
        probability of using a ground truth frame instead of a previous one.
    sequence_length : int

    Returns
    -------
    These batched tensors:
     - frames(float): input heatmaps and frames
     - deleted_frames(int): which input heatmaps have been deleted
     - use_gt(int): whether to use ground truth heatmaps or the previous state
     - labels(float): the output heatmap(s)
    """
    if total_clear_probability + clear_probability > 1:
        raise ValueError(f"total_clear_probability + clear_probability > 1, with total_clear_probability = {total_clear_probability} and clear_probability = {clear_probability}")

    frames, labels = default_collate(batch)
    deleted_frames = torch.zeros(len(batch))

    x = frames.clone()
    for i in range(len(batch)):
        if torch.rand(1) < total_clear_probability:
           # clear all previous heatmaps with a probability of total_clear_probability
            x[i, :sequence_length-1] = torch.zeros(sequence_length-1, x.shape[2], x.shape[3])
            deleted_frames[i] = sequence_length-1
        elif torch.rand(1) < clear_probability/(1-total_clear_probability):
           # if not, clear previous heatmaps in increasing order witn a uniform distribution with probability clear_probability
            to_delete = torch.randint(low=1, high=sequence_length, size=(1,))
            x[i, :to_delete] = torch.zeros(to_delete, x.shape[2], x.shape[3])
            deleted_frames[i] = to_delete

    use_gt = Bernoulli(ground_truth_probability*torch.ones(len(batch), sequence_length-1)).sample()
    return x, deleted_frames, use_gt, labels


class BatchSamplerRNN(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        if drop_last:
            self.num_batches = len(data_source) // batch_size
        else:
            self.num_batches = (len(data_source) + batch_size - 1) // batch_size

    def __iter__(self):
        i = 0
        for batch_idx in range(self.num_batches):
            batch = []
            for element_idx in range(self.batch_size):
                i+=1
                if i > len(self.data_source):
                    break
                batch.append(batch_idx + element_idx*self.num_batches)
            yield batch

    def __len__(self):
        return self.num_batches


class ConfigBase():
    """Base class for Config()"""
    _checkpoint_folder = None
    def get_model(self):
        raise NotImplementedError


def _get_standard_dataset(root, train_configuration, is_rnn):
    config = train_configuration.Config()

    if hasattr(config, '_grayscale'):
        grayscale = config._grayscale
    else:
        grayscale = False

    dataset_params = dict(root=root,
                          image_size=config._image_size,
                          sigma=5,
                          sequence_length=config._sequence_length,
                          one_output_frame=config._one_output_frame,
                          drop_duplicate_frames=False,
                          transform=ToTensor(),
                          target_transform=ToTensor(),
                          grayscale=grayscale)

    if is_rnn:
        return VideoDatasetRNN(**dataset_params)
    return VideoDataset(**dataset_params)


def get_standard_test_dataset(train_configuration, name='test_standard', is_rnn = False):
    return _get_standard_dataset(f'../datasets/{name}', train_configuration, is_rnn)


def get_debug_dataset(train_configuration, is_rnn = False):
    return _get_standard_dataset('../datasets/debug', train_configuration, is_rnn)
