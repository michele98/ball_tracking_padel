import os
import time

import torch
import torch.utils.data
import torch.optim as optim
from utils.storage import *

from torch.optim.lr_scheduler import OneCycleLR, CyclicLR

"""Adapted from https://github.com/EnricoPittini/denoising-diffusion-models"""


def train_one_epoch(net : torch.nn.Module,
                    dataloader_train : torch.utils.data.DataLoader,
                    loss_function : torch.nn.Module,
                    optimizer : torch.optim.Optimizer = None,
                    scheduler : torch.optim.lr_scheduler._LRScheduler = None,
                    device : str = 'cpu',
                    scaler = torch.cuda.amp.GradScaler(),
                    prefix : str = ''):
    """Train the given model for one epoch, over the given dataset

    Parameters
    ----------
    net : torch.nn.Module
    dataloader_train : torch.utils.data.DataLoader
    loss_function : torch.nn.Module
    optimizer : torch.optim.Optimizer, optional
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        If given, it is assumed that the scheduler step must be performed after each batch, and not after each epoch.
        If the scheduler step must be performed after each epoch, do not specify any scheduler.
        PyTorch schedulers for which the update must be performed after each batch: `OneCycleLR`, `CyclicLR`.
    device : str, optional
    scaler : torch.cuda.amp.GradScaler(), optional
        The scaler for using 16 bits precision
    prefix : str, optional
        String to append at the beginning of the output information, by default ''

    Returns
    -------
    loss_train : np.array
        Single scalar value, representing the computed loss value over the whole dataset in that epoch
    """

    net.train()         # set model to training mode

    tot_error=0
    tot_images=0
    start_time = time.time()

    for batch_idx, data in enumerate(dataloader_train):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):  # 16 bit precision (for using less memory)
            # Compute prediction (forward input in the model)
            outputs = net(inputs)

            # Compute prediction error with the loss function
            error = loss_function(outputs, labels)

        # Backpropagation
        #net.zero_grad()
        #error.backward()
        scaler.scale(error).backward()

        # Optimizer step
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        tot_error += error*len(labels)      # weighted average
        tot_images += len(labels)

        loss = tot_error/tot_images

        # Update of the LR, according to the given scheduler (update after each batch)
        if scheduler is not None:
            # print('UPDATE BATCH')
            scheduler.step()

        epoch_time = time.time() - start_time
        batch_time = epoch_time/(batch_idx+1)

        print(prefix + f"{batch_idx+1}/{len(dataloader_train)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {loss:.3g}".ljust(80), end = '\r')

    print(prefix + f"{batch_idx+1}/{len(dataloader_train)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {loss:.3g}".ljust(80))
    loss_np = (loss).detach().cpu().numpy()

    return loss_np


def validate(net : torch.nn.Module,
             dataloader_val : torch.utils.data.DataLoader,
             loss_function : torch.nn.Module,
             device : str = 'cpu',
             prefix=''):
    """Evaluate the given model on the given dataset.

    Parameters
    ----------
    net : torch.nn.Module
    dataloader_val : torch.utils.data.DataLoader
    loss_function : torch.nn.Module
        Metric to use for the evaluation
    device : str, optional
    prefix : str, optional
        String to append at the beginning of the output information, by default ''

    Returns
    -------
    loss_val : np.array
        Single scalar value, representing the computed loss value over the whole dataset
    """
    net.eval()         # set model to evaluation mode

    tot_error=0
    tot_images=0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader_val):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                
                # Compute prediction (forward input in the model)
                outputs = net(inputs)

                # Compute prediction error with the loss function
                error = loss_function(outputs, labels)

            tot_error += error*len(labels)      # weighted average
            tot_images += len(labels)

            loss = tot_error/tot_images

            epoch_time = time.time() - start_time
            batch_time = epoch_time/(batch_idx+1)

            print(prefix + f'{batch_idx+1}/{len(dataloader_val)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, loss: {loss:.3g}'.ljust(80), end = '\r')

    print(prefix + f'{batch_idx+1}/{len(dataloader_val)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, loss: {loss:.3g}'.ljust(80))
    loss_np = (loss).detach().cpu().numpy()

    return loss_np


def train_model(net : torch.nn.Module,
                dataloader_train : torch.utils.data.DataLoader,
                dataloader_val : torch.utils.data.DataLoader,
                loss_function : torch.nn.Module,
                epochs : int,
                optimizer : torch.optim.Optimizer = None,
                scheduler : torch.optim.lr_scheduler._LRScheduler = None,
                device : torch.device = None,
                checkpoint_folder : str = None,
                additional_info : dict = {},
                checkpoint_step : int = 1,
                clear_previous_checkpoints=True,
                keep_best=True,
                verbose=False):
    """Training loop.

    Parameters
    ----------
    net : torch.nn.Module
        the model to train
    dataloader_train : torch.utils.data.DataLoader
    dataloader_val : torch.utils.data.DataLoader
    loss_function : torch.nn.Module
    epochs : int
    optimizer : torch.optim.Optimizer, optional
        by default Adam.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        learning rate scheduler, by default None.
        The update of the LR is performed either after each epoch (standard update) or after each batch (only for
        `OneCycleLR` and `CyclicLR`).
    device : torch.device, optional
        cpu or cuda, by default cpu.
    checkpoint_folder : str, optional
        folder into which to save the training checkpoints. If not provided, no checkpoints are
        created.
    additional_info : dict, optional
        additional info to save alongside the checkpoint information.
    checkpoint_step : int, optional
        every how many epochs a checkpoint is created, by default 1.
    clear_previous_checkpoints : bool, optional
        if set to True, the previous checkpoints are deleted. The epoch number is appended at the
        end of the filename provided by ``checkpoint_filename``. By default False.
    keep_best : bool, optional
        if set to True, keeps also the checkpoint with the best loss. Has an effect only if
        ``clear_previous_checkpoints`` is set to True.
    verbose : bool, optional
        if true, prints each time chekpoints are created.

    Returns
    -------
    dict
        the checkpoint dictionary.
    """

    # -------------------- SETUP -------------------- #
    if device is None:
        device = torch.device('cpu')
    net.to(device)
    print(f"Device: {device}")

    if optimizer is None:
        optimizer = optim.Adam(net.parameters())
    else:
        optimizer = optimizer(params=net.parameters())

    if scheduler is not None:
        scheduler = scheduler(optimizer=optimizer)
        
        # Understand whether the LR scheduler must be update after each epoch (classic update) or after each batch (only for
        # two schedulers).
        scheduler_update_each_batch = False
        if type(scheduler)==OneCycleLR or type(scheduler)==CyclicLR:
            scheduler_update_each_batch = True

    save_checkpoints = checkpoint_folder is not None

    starting_epoch = 0
    loss_history = []
    loss_history_val = []

    # The scaler for using 16 bits precision
    scaler = torch.cuda.amp.GradScaler()

    # resume from previous checkpoint
    if checkpoint_folder is not None:
        if os.path.exists(checkpoint_folder):
            checkpoint = load_checkpoint(checkpoint_folder=checkpoint_folder, net=net, optimizer=optimizer,
                                         scheduler=scheduler)
            if checkpoint is not None:
                starting_epoch, net, optimizer, scheduler, loss_history, loss_history_val, additional_info = checkpoint
                print("Checkpoint loaded.")
        else:
            os.makedirs(checkpoint_folder)
            print(f"Created checkpoint folder {checkpoint_folder}")

    if not verbose: print(" ")

    # -------------------- TRAINING -------------------- #
    # loop for every epoch (training + evaluation)
    for i, epoch in enumerate(range(starting_epoch, epochs+starting_epoch)):
        if verbose: print(" ")
        print(f'Epoch: {epoch+1}/{epochs+starting_epoch}')

        # Training epoch
        train_loss = train_one_epoch(net=net,
                                      dataloader_train=dataloader_train,
                                      loss_function=loss_function,
                                      optimizer=optimizer,
                                      # Specify the scheduler to the train epoch only if a scheduler exists and if the LR
                                      # update must be performed after each batch (and not after each epoch)
                                      scheduler=None if (scheduler is None or not scheduler_update_each_batch) else scheduler,
                                      device=device,
                                      scaler=scaler,
                                      prefix='\tTrain ')
        loss_history.append(train_loss)

        # Validation
        val_loss = validate(net=net,
                            dataloader_val=dataloader_val,
                            loss_function=loss_function,
                            device=device,
                            prefix='\tVal ')
        loss_history_val.append(val_loss)

        # Update of the LR according to the scheduler (if the update must be performed after each epoch)
        if scheduler is not None and not scheduler_update_each_batch:
            #print('UPDATE EPOCH')
            scheduler.step()

        # create checkpoint dictionary
        if i%checkpoint_step == 0:
            checkpoint_dict = create_checkpoint_dict(net=net,
                                                     epoch=epoch+1,
                                                     optimizer=optimizer,
                                                     scheduler=scheduler,
                                                     loss_history=loss_history,
                                                     loss_history_val=loss_history_val,
                                                     additional_info=additional_info)

            # save checkpoint dict if filename is provided
            if save_checkpoints:
                save_checkpoint(checkpoint_dict, checkpoint_folder, clear_previous_checkpoints=clear_previous_checkpoints,
                                keep_best=keep_best, verbose=verbose)

    print('\nTraining done.')

    return checkpoint_dict
