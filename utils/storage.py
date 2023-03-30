import os
import numpy as np

import torch
import torch.utils.data

"""Taken from https://github.com/EnricoPittini/denoising-diffusion-models"""


def create_checkpoint_dict(net : torch.nn.Module,
                           epoch : int,
                           optimizer : torch.optim.Optimizer,
                           scheduler : torch.optim.lr_scheduler._LRScheduler,
                           loss_history : list,
                           loss_history_val : list,
                           additional_info : dict = {}):
    """Get the training checkpoint dictionary.

    Parameters
    ----------
    net : torch.nn.Module
    epoch : int
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler
    loss_history : list
    loss_history_val : list
    additional_info : dict, optional

    Returns
    -------
    checkpoint_dict : dict
        Dictionary containing the checkpoint information. Namely:
        - epoch
        - model_state_dict
        - optimizer_state_dict
        - scheduler_state_dict
        - loss_history
        - loss_history_val
        - additional_info
    """
    additional_info['model'] = str(type(net))
    additional_info['optimizer'] = str(type(optimizer))
    additional_info['scheduler'] = str(type(scheduler))

    return {'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'loss_history': loss_history,
            'loss_history_val': loss_history_val,
            'additional_info': additional_info
           }


def save_checkpoint(checkpoint_dict, checkpoint_folder, clear_previous_checkpoints=True, keep_best=True, verbose=False):
    """Save the given checkpoint dictionary into the specified checkpoint folder.

    Parameters
    ----------
    checkpoint_dict : dict
        Dictionary containing the checkpoint information. Namely:
        - epoch
        - model_state_dict
        - optimizer_state_dict
        - scheduler_state_dict
        - loss_history
        - loss_history_val
        - additional_info
    checkpoint_folder : str
        Folder into which storing the checkpoint
    clear_previous_checkpoints : bool, optional
        Whether to clear the previous saved checkpoints or not, by default True
    keep_best : bool, optional
        Whether to keep only the best checkpoint or not, by default True
    verbose : bool, optional
        Whether to be verbose or not, by default False
    """
    filename = 'checkpoint_' + f"{checkpoint_dict['epoch']}".zfill(4)

    # put best flag
    loss_history_val = checkpoint_dict['loss_history_val']
    clean_history = [element for element in loss_history_val if not np.isnan(element)]
    if (
        len(clean_history)==0 or (
            not np.isnan(loss_history_val[-1]) and
            abs(loss_history_val[-1]-min(clean_history))<1e-8
            )
        ):
        filename += '_best'

    filename += '.ckpt'
    filepath = os.path.join(checkpoint_folder, filename)
    torch.save(checkpoint_dict, filepath)

    # save loss history train
    np.savetxt(os.path.join(checkpoint_folder, 'loss_history.csv'), checkpoint_dict['loss_history'], delimiter=',')
    # save loss history val
    np.savetxt(os.path.join(checkpoint_folder, 'loss_history_val.csv'), checkpoint_dict['loss_history_val'], delimiter=',')

    if verbose: print(f"Checkpoint saved: {filepath}.")

    if clear_previous_checkpoints:
        _clear_checkpoint_folder(checkpoint_folder, keep_best)
        if verbose: print('Cleared previous checkpoints.')


def _clear_checkpoint_folder(checkpoint_folder, keep_best):
    checkpoints = sorted([i for i in os.listdir(checkpoint_folder) if i != 'loss_history.csv' and i != 'loss_history_val.csv' and '.ckpt' in i])

    best_found = '_best' in checkpoints[-1]

    for c in reversed(checkpoints[:-1]):
        if best_found:
            os.remove(os.path.join(checkpoint_folder, c))
        else:
            if '_best' in c and keep_best:
                best_found = True
            else:
                os.remove(os.path.join(checkpoint_folder, c))


def load_checkpoint_dict(checkpoint_folder : str):
    """Load the checkpoint dictionary from the specified checkpoint folder.

    Parameters
    ----------
    checkpoint_folder : str
        folder containing the checkpoint file.

    Returns
    -------
    dict
        Dictionary containing the checkpoint information. Namely:
        - epoch
        - model_state_dict
        - optimizer_state_dict
        - scheduler_state_dict
        - loss_history
        - loss_history_val
        - additional_info

    Raises
    ------
    FileNotFoundError
        if ``checkpoint_folder`` does not exist.
    """
    if not os.path.exists(checkpoint_folder):
        raise FileNotFoundError(f"The folder {checkpoint_folder} does not exist.")

    if len([i for i in os.listdir(checkpoint_folder) if i[-5:]=='.ckpt']) == 0:
        print(f"No checkpoint found in {checkpoint_folder}, using default initialization.")
        return None

    filename = [i for i in os.listdir(checkpoint_folder) if i != 'loss_history.csv' and i != 'loss_history_val.csv' and '.ckpt' in i][-1]
    filepath = os.path.join(checkpoint_folder, filename)

    print(f"Loading checkpoint: {filepath}")
    checkpoint_dict = torch.load(filepath, map_location=torch.device('cpu'))

    return checkpoint_dict


def load_checkpoint(checkpoint_folder : str,
                    net : torch.nn.Module,
                    optimizer : torch.optim.Optimizer,
                    scheduler : torch.optim.lr_scheduler._LRScheduler):
    """Load training status from a checkpoint folder.

    Parameters
    ----------
    checkpoint_folder : str
        folder containing the checkpoint file.
    net : torch.nn.Module
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler

    Returns
    -------
    epoch : int
    net : torch.nn.Module
        the model with the loaded ``state_dict``.
    optimizer : torch.optim.Optimizer
        the optimizer with the loaded ``state_dict``.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        the scheduler with the loaded ``state_dict``
    loss_history : list
    loss_history_val : list
    additional_info : dict
    None if ``checkpoint_folder`` is empty.

    Raises
    ------
    FileNotFoundError
        if ``checkpoint_folder`` does not exist.
    """
    checkpoint = load_checkpoint_dict(checkpoint_folder)
    if checkpoint is None:
        return None

    epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    loss_history = checkpoint['loss_history']
    loss_history_val = checkpoint['loss_history_val']
    additional_info = checkpoint['additional_info']

    return epoch, net, optimizer, scheduler, loss_history, loss_history_val, additional_info


def load_weights(net : torch.nn.Module, checkpoint_filename : str):
    """Load weights from a checkpoint. The model is automatically set to eval mode.

    Parameters
    ----------
    net : torch.nn.Module
    checkpoint_filename : str
    """
    if not os.path.exists(checkpoint_filename):
        raise FileNotFoundError(f"The file {checkpoint_filename} does not exist.")

    checkpoint = torch.load(checkpoint_filename)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
