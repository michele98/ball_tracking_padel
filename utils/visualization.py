import os
import numpy as np
import matplotlib.pyplot as plt


def show_loss_history(module):
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

    fig, ax = plt.subplots()
    ax.set_yscale('log')

    ax.plot(train_loss, label='train loss')
    ax.plot(val_loss, label='val loss')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.legend()

    fig.tight_layout()

    plt.show()
