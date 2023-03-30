import torch
from torch.utils.data._utils.collate import default_collate
from torch.distributions.bernoulli import Bernoulli


def collate_fn_rnn(batch,
                   total_clear_probability: float,
                   clear_probability: float,
                   ground_truth_probability: list[float],
                   sequence_length: int):
    """Base collate function to pass to a DataLoader for TrackNetV2RNN.
    When passing to a DataLoader, fix `clear_probability` and `sequence_length` using functools.partial

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


class ConfigBase():
    """Base class for Config()"""
    _checkpoint_folder = None
    def get_model(self):
        raise NotImplementedError
