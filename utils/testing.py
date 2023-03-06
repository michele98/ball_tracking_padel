import os
import cv2
import time
import numpy as np
import pandas as pd

import torch
import torch.utils.data


def get_maximum_coordinates(heatmaps):
    y, x = np.nonzero(heatmaps == np.max(heatmaps))
    return x[0], y[0]


def compute_positions(net : torch.nn.Module,
                      data_loader : torch.utils.data.DataLoader,
                      device : str = 'cpu',
                      heatmaps_folder : str = None,
                      prefix : str = '',):
    """Compute the position of the ball for all the frames in the dataloader.

    Parameters
    ----------
    net : torch.nn.Module
        the model to evaluate
    data_loader : torch.utils.data.DataLoader
        dataloader of the test set
    device : torch.device, optional
        cpu or cuda, by default cpu
    heatmaps_folder : string, optional
        if provided, saves the heatmaps of each frame

    Returns
    -------
    true_positions : np.ndarray
        list of ground truth positions in (y, x) pixel coordinates
    predicted_positions : np.ndarray
        list of predicted positions in (y, x) pixel coordinates
    min_heatmap_values : np.ndarray
        minimum values of the output activation
    max_heatmap_values : np.ndarray
        maximum values of the output activation
    """
    if heatmaps_folder is not None:
        if not os.path.exists(heatmaps_folder):
            os.makedirs(heatmaps_folder)

    net.eval() # set the model in evaluation mode

    true_positions = []
    predicted_positions = []
    min_heatmap_values = []
    max_heatmap_values = []

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            inputs = data[0].to(device)
            labels = data[1].numpy()

            outputs = net(inputs).to('cpu').numpy()

            for i, (label, output) in enumerate(zip(labels, outputs)):
                true_positions.append(get_maximum_coordinates(label[-1]))
                predicted_positions.append(get_maximum_coordinates(output[-1]))
                min_heatmap_values.append(output[0].min())
                max_heatmap_values.append(output[0].max())

                if heatmaps_folder is not None:
                    filepath = os.path.join(heatmaps_folder, f"{batch_idx*data_loader.batch_size + i}".zfill(6) + '.png')
                    # normalize heatmap between 0 and 1
                    heatmap = (output[0]-output[0].min()) / (output[0].max()-output[0].min())
                    cv2.imwrite(filepath, np.asarray(heatmap*255, dtype=np.uint8))

            epoch_time = time.time() - start_time
            batch_time = epoch_time/(batch_idx+1)
            print(prefix + f'{batch_idx+1}/{len(data_loader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step'.ljust(80), end = '\r')
    print(prefix + f'{batch_idx+1}/{len(data_loader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step'.ljust(80))

    return np.array(true_positions), np.array(predicted_positions), np.array(min_heatmap_values), np.array(max_heatmap_values)


def frame_generator(filename, start_frame=None, stop_frame=None, verbose=True):
    """Generator that yields frames from a video.

    Parameters
    ----------
    filename : string
        name of the video file.
    start_frame : int, optional
        starting frame from which to read the video, by default 0
    stop_frame : int, optional
        final frame from which to read the video, by default the final frame
    verbose : bool, optional
        by default True

    Yields
    ------
    array
        the current video frame. The channel order is RGB.

    Raises
    ------
    FileNotFoundError
        if the video file does not exist
    ValueError
        if start_frame >= stop_frame
    """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise FileNotFoundError(f'Video file {filename} not found!')

    if start_frame is None:
        start_frame = 0

    if stop_frame is None:
        stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame >= stop_frame:
        raise ValueError("the starting frame must be smaller than the stopping frame.")

    current_frame = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    ret, frame = cap.read()
    for i in range(start_frame, stop_frame):
        if verbose: print(f"writing frame {i-start_frame+1} of {stop_frame-start_frame}".ljust(80), end = '\r')
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ret, frame = cap.read()
        if not ret:
            if verbose: print("Finished prematurely".ljust(80))
            break
    if verbose: print(f"writing frame {i-start_frame+1} of {stop_frame-start_frame}".ljust(80))
    if verbose: print("Finished frames.")
    cap.release()


def annotate_frame(frame, predicted_position, true_position=None):
    """Put dots on the predicted (red) and true (green) ball positions.

    Parameters
    ----------
    frame : array
        frame to annotate
    predicted_position : tuple of float
        predicted ball position in pixel coordinates. The coordinate order is `(x, y)`
    true_position : tuple of float, optional
        true ball position in pixel coordinates. The coordinate order is `(x, y)`

    Returns
    -------
    annotated_frame : array
        it has the same shape as `frame`.
    """
    annotated_frame = frame.copy()

    if true_position is not None:
        annotated_frame = cv2.circle(annotated_frame,
                                     center=true_position,
                                     radius=5,
                                     color=(0, 255, 0),
                                     thickness=cv2.FILLED)

    annotated_frame = cv2.circle(annotated_frame,
                                 center=predicted_position,
                                 radius=5,
                                 color=(255, 0, 0),
                                 thickness=cv2.FILLED)

    return cv2.addWeighted(annotated_frame, 0.6, frame, 0.4, 0)


def save_labeled_video(filename_src : str,
                       filename_dst : str,
                       position_df : pd.DataFrame,
                       show_ground_truth : bool = False,
                       heatmaps_folder : str = None,
                       start_frame : int = None,
                       stop_frame : int = None,
                       fps : int = None):
    """Save the video with the annotated ball position.

    Parameters
    ----------
    filename_src : string
        name of the source video file.
    filename_dst : string
        name of the destination video file.
    position_df : pd.DataFrame
        dataframe with the position data.
        Must contain the following columns:
         - `frame_num`
         - `x_pred`
         - `y_pred`
         - `x_true` (only if `show_ground_truth` is True)
         - `y_true` (only if `show_ground_truth` is True)
    show_ground_truth : bool, optional
        by default False
    heatmaps_folder : string, optional
        if provided
    start_frame : int, optional
        if provided, starts the rendering after this many frames.
        By default the first annotated frame in `position_df`.
    stop_frame : int, optional
        if provided, stops the rendering after this many frames.
        The count starts from frame 0 of the original video, and it
        needs to be greater than `start_frame`.
        By default the last annotated frame in `position_df`.
    fps : int, optional
        frames per second of the video, by default the one of the source video file.
    """
    if start_frame is None:
        start_frame = position_df['frame_num'].min()
    if stop_frame is None:
        stop_frame = position_df['frame_num'].max() - start_frame
    if fps is None:
        cap = cv2.VideoCapture(filename_src)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    frame_gen = frame_generator(filename_src, start_frame, start_frame+stop_frame) #initialize frame generator
    first_frame = next(frame_gen)

    # create the VideoWriter object
    # and set the resolution of the output video as the one of the input video
    h, w = first_frame.shape[0], first_frame.shape[1]

    out = cv2.VideoWriter(filename=filename_dst,
                          fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                          fps=fps,
                          frameSize=(w, h))

    for i, frame in enumerate(frame_gen):
        frame_index = position_df.loc[position_df['frame_num']==i+start_frame-1].index
        if len(frame_index) == 1:
            x_pred = int(position_df['x_pred'][frame_index]*frame.shape[1])
            y_pred = int(position_df['y_pred'][frame_index]*frame.shape[0])
            predicted_position = (x_pred, y_pred)

            if show_ground_truth:
                x_true = int(position_df['x_true'][frame_index]*frame.shape[1])
                y_true = int(position_df['y_true'][frame_index]*frame.shape[0])
                true_position = (x_true, y_true)
            else:
                true_position = None

            # add heatmap visualization
            if heatmaps_folder is not None:
                heatmap = cv2.imread(os.path.join(heatmaps_folder, f"{frame_index.values[0]}".zfill(6)+'.png'))
                heatmap = cv2.resize(heatmap, (w, h))
                frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

            frame = annotate_frame(frame, predicted_position, true_position)

        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print('done.')
