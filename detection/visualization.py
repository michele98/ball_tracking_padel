import os
import numpy as np
import matplotlib.pyplot as plt


def get_loss_history(module):
    """Get train loss and validation loss of the given module"""
    try:
        train_loss = np.loadtxt(os.path.join(module.Config()._checkpoint_folder, "loss_history.csv"))
        val_loss = np.loadtxt(os.path.join(module.Config()._checkpoint_folder, "loss_history_val.csv"))
    except OSError:
        phases = ['phase_1_0', 'phase_1_1', 'phase_2_0', 'phase_2_1', 'phase_3_0', 'phase_3_1']

        train_loss = []
        val_loss = []

        for phase in phases:
            folder = os.path.join(module.Config()._checkpoint_folder, phase)

            train_loss.append(np.loadtxt(os.path.join(folder, "loss_history.csv")))
            val_loss.append(np.loadtxt(os.path.join(folder, "loss_history_val.csv")))

        train_loss = np.concatenate(train_loss)
        val_loss = np.concatenate(val_loss)
    return train_loss, val_loss


def show_loss_history(module, ax=None, epoch_range=None):
    """Plot the loss history of the given module

    Parameters
    ----------
    module : _type_
        _description_
    ax : matplotlib.axes.Axes, optional
        _description_, by default None
    epoch_range : tuple of int, shape (2,), optional
        epoch range to plot, including both ends.
        If not provided, the whole history is plotted

    Returns
    -------
    matplotlib.axes.Axes
    """
    train_loss, val_loss = get_loss_history(module)

    if epoch_range is not None:
        train_loss = train_loss[epoch_range[0]-1:epoch_range[1]]
        val_loss = val_loss[epoch_range[0]-1:epoch_range[1]]

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_yscale('log')

    ax.plot(range(1, len(train_loss)+1), train_loss, '.-', label='train loss')
    ax.plot(range(1, len(train_loss)+1), val_loss, '.-', label='val loss')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.legend()

    return ax
