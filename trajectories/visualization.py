import gc
import io

import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from detection.testing import annotate_frame, frame_generator

from trajectories.data_reading import get_candidates, get_heatmap, get_video_source
from trajectories.fitting import fit_trajectories
from trajectories.filtering import (build_path_mapping, build_trajectory_graph,
                                    find_next_nodes, find_prev_nodes,
                                    find_shortest_paths)


def figure_to_array(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im[:,:,:3]


def visualize_trajectory_graph(trajectory_graph: nx.DiGraph, first_node: int = None, last_node: int = None, ax:plt.Axes=None):
    nodes_list = sorted(list(trajectory_graph.nodes()))

    if first_node is None:
        first_node_index = np.random.randint(len(nodes_list) - 20)
        first_node = nodes_list[first_node_index]
    else:
        nodes_before = [n for n in nodes_list if n<=first_node]
        if len(nodes_before) > 0:
            first_node = nodes_before[-1]
        else:
            nodes_after = [n for n in nodes_list if n>first_node]
            first_node = nodes_after[0]

    if last_node is None:
        last_node = nodes_list[nodes_list.index(first_node) + 20]

    G = trajectory_graph.subgraph(range(first_node, last_node+1))

    if ax is None:
        w, h, dpi = 1800, 600, 100
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    else:
        fig = ax.get_figure()

    # Create a linear layout
    pos = {node: (i, 0) for i, node in enumerate(sorted(G.nodes()))}
    print(len(pos))

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, linewidths=2)

    # Draw the curved edges
    for u, v, weight in G.edges.data('weight'):
        edge_color = 'r' if weight > 0 else 'k'
        # d = -0.2 * np.log(v-u)  # Control the curvature of the edges
        x = nodes_list.index(v) - nodes_list.index(u)
        d = 0.5*(1-np.exp(-0.4*(x-1)))  # Control the curvature of the edges
        d = d if u%2 == 1 else -d
        xs, ys = pos[u]
        xt, yt = pos[v]
        xc = (xs + xt) / 2
        yc = (ys + yt) / 2
        xc += d * (yt - ys)
        yc += d * (xs - xt)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=edge_color, width=1, alpha=1, connectionstyle=f'arc3,rad={d}', arrowstyle='-|>', arrowsize=10)

    # Add labels to the nodes
    nx.draw_networkx_labels(G, pos, font_color='k')

    # Set the x-axis limits to include the nodes
    ax.set_xlim(-0.5, len(G.nodes())-0.5)
    # Set the y-axis limits
    ax.set_ylim([-1, 1])

    fig.tight_layout()

    # Show the graph
    ax.set_axis_off()

    return ax


def show_single_trajectory(fitting_info,
                           candidates,
                           k_seed,
                           frame=None,
                           heatmap=None,
                           k_min = None,
                           i_min = None,
                           k_max = None,
                           i_max = None,
                           ax=None,
                           show_outside_range=False,
                           display='k_min stats',
                           stat_frame = None,
                           annotate=True,
                           show_fitting_points=False,
                           trajectory_color='y',
                           fontsize=12,
                           dpi=100,
                           alpha=0.8,
                           line_style='.-',
                           verbose=True,
                           **kwargs):
    if ax is None:
        w, h = 1280, 720
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    else:
        fig = ax.figure

    # display frame
    if frame is not None:
        ax.imshow(frame, zorder=-200)
        if heatmap is not None:
            ax.imshow(cv2.resize(heatmap, (frame.shape[1], frame.shape[0])), zorder=-199, alpha=0.5, cmap='gray', vmin=0, vmax=1)
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)

    # find whether trajectory is found
    trajectories_info = fitting_info['trajectories']
    starting_frame = trajectories_info[0]['k_seed']

    trajectory_info = None
    exists_trajectory = False
    i_seed = 0
    if k_seed is not None:
        # offset k_seed by starting frame and get trajectory_info
        k_seed -= starting_frame
        trajectory_info = trajectories_info[k_seed]
        # get i_seed
        i_seed = trajectory_info['i_seed']
        exists_trajectory = trajectory_info['found_trajectory']
    else:
        k_seed = starting_frame


    # find k_min, k_mid and and k_max
    if exists_trajectory:
        if k_min is None:
            k_min = trajectory_info['k_min']
            i_min = trajectory_info['i_min']
        elif i_min is None:
            raise ValueError('You must pass both k_min and i_min')

        k_min -= starting_frame

        if k_max is None:
            k_max = trajectory_info['k_max']
            i_max = trajectory_info['i_mid']
        elif i_max is None:
            raise ValueError('You must pass both k_max and i_max')

        k_max -= starting_frame

        k_mid = trajectory_info['k_mid'] - starting_frame
        i_mid = trajectory_info['i_mid']

    # display trajectory statistics
    if annotate and display is not None and ('parameters' in display or
                                             'params' in display or
                                             'p' in display or
                                             's' in display or
                                             'stats' in display or
                                             'all' in display):
        s = 7

        ann = f"{stat_frame}".ljust(s+2) if stat_frame is not None else " "*(s+2)
        ann += f"x" + " "*(s-1) + "y" + " "*(s-3) + "|.|"
        ann += "\n" + "\u2014"*(s*3+3) + "\n"

        if exists_trajectory:
            a = trajectory_info['a']
            v0 = trajectory_info['v']

            ann += "v0 " + f"{v0[1]:.2f}".rjust(s) + f"{-v0[0]:.2f}".rjust(s) + f"{np.linalg.norm(v0):.2f}".rjust(s)
            ann += "\n"
            ann += "a  " + f"{a[1]:.2f}".rjust(s) + f"{-a[0]:.2f}".rjust(s) + f"{np.linalg.norm(a):.2f}".rjust(s)
            if stat_frame is not None:
                v = a*(stat_frame - k_min - starting_frame) + v0
                ann += "\n"
                ann += "v  " + f"{v[1]:.2f}".rjust(s) + f"{-v[0]:.2f}".rjust(s) + f"{np.linalg.norm(v):.2f}".rjust(s)
        else:
            ann += "v0 " + f"---".rjust(s) + f"---".rjust(s) + f"---".rjust(s)
            ann += "\n"
            ann += "a  " + f"---".rjust(s) + f"---".rjust(s) + f"---".rjust(s)
            if stat_frame is not None:
                ann += "\n"
                ann += "v  " + f"---".rjust(s) + f"---".rjust(s) + f"---".rjust(s)

        bbox = {'boxstyle': 'square,pad=0.7',
                'facecolor': '#232323',
                'edgecolor': '#FFFFFF',
                'linewidth': 0.5,
                'alpha': 0.85}
        font = FontProperties(family='monospace', size=fontsize)
        ax.annotate(ann, [40, 40], fontproperties=font, bbox=bbox, color='#FFFFFF', va='top')

    # if the trajectory is not found, return the axes as they are
    if not exists_trajectory:
        if verbose:
            print(f'No fitted trajectory for frame {k_seed}')
        return ax

    # display trajectory
    trajectory = trajectory_info['trajectory']

    # trajectory
    k = np.arange(len(trajectory)) + k_seed - (len(trajectory)-1)//2
    if show_outside_range:
        ax.plot(trajectory[k<=k_min,1],trajectory[k<=k_min,0], f'{trajectory_color}{line_style}', zorder=-1, alpha=alpha/4, **kwargs)
        ax.plot(trajectory[k>=k_max,1],trajectory[k>=k_max,0], f'{trajectory_color}{line_style}', zorder=-1, alpha=alpha/4, **kwargs)
    mask = np.logical_and(k>=k_min, k<=k_max)
    ax.plot(trajectory[mask,1],trajectory[mask,0], f'{trajectory_color}{line_style}', zorder=-1, alpha=alpha, **kwargs)

    if display is not None:
        bbox = {'boxstyle': 'round',
                'facecolor': trajectory_color,
                'edgecolor': 'none',
                'alpha': 0.4}
        font = FontProperties(family='monospace', weight='bold', size=fontsize)

        if 'all' in display:
            display = ['k_min', 'k_mid', 'k_max', 'k_seed']

        if 'k_seed' in display:
            if annotate:
                ax.annotate(k_seed + starting_frame, [candidates[k_seed, i_seed, 1], candidates[k_seed, i_seed, 0]], fontproperties=font, bbox=bbox, color='k')
            if show_fitting_points:
                ax.scatter(candidates[k_seed, i_seed, 1], candidates[k_seed, i_seed, 0], c='k', s=5)

        if 'k_min' in display:
            if annotate:
                ax.annotate(k_min + starting_frame, [candidates[k_min, i_min, 1], candidates[k_min, i_min, 0]], fontproperties=font, bbox=bbox, color='w')
            if show_fitting_points:
                ax.scatter(candidates[k_min, i_min, 1], candidates[k_min, i_min, 0], c='w', marker='^')

        if 'k_mid' in display:
            if annotate:
                ax.annotate(k_mid + starting_frame, [candidates[k_mid, i_mid, 1], candidates[k_mid, i_mid, 0]], fontproperties=font, bbox=bbox, color='w')
            if show_fitting_points:
                ax.scatter(candidates[k_mid, i_mid, 1], candidates[k_mid, i_mid, 0], c='w')

        if 'k_max' in display:
            if annotate:
                ax.annotate(k_max + starting_frame, [candidates[k_max, i_max, 1], candidates[k_max, i_max, 0]], fontproperties=font, bbox=bbox, color='w')
            if show_fitting_points:
                ax.scatter(candidates[k_max, i_max, 1], candidates[k_max, i_max, 0], c='w', marker='s')

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return ax


def show_trajectory_sequence(fitting_info: dict,
                             candidates: np.ndarray,
                             sequence: list,
                             frame: np.ndarray = None,
                             heatmap: np.ndarray = None,
                             ax=None,
                             dpi=100,
                             link_trajectories=True,
                             colors=None,
                             **kwargs):
    """Show trajectories superimposed on one frame.

    Parameters
    ----------
    fitting_info : dict
        _description_
    candidates : np.ndarray
        detection candidates
    sequence : list of int
        trajectory sequence. Usually it is the shortest path found with djikstra
    frame : np.ndarray, optional
        frame to show
    ax : plt.axes.Axes, optional
    """
    if ax is None:
        w, h = 1280, 720
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    else:
        fig = ax.get_figure()

    if colors is None:
        colors = ['r', 'w', 'g', 'y']

    trajectories_info = fitting_info['trajectories']

    seed_sequence = [t['k_seed'] for t in trajectories_info]
    for i, node in enumerate(sequence):
        k_max = None
        i_max = None
        if link_trajectories and i < len(sequence)-1:
            next_node = sequence[i+1]

            k_min = trajectories_info[seed_sequence.index([node])]['k_min']
            k_max = trajectories_info[seed_sequence.index([node])]['k_max']
            k_min_next = trajectories_info[seed_sequence.index([next_node])]['k_min']

            if k_min >= k_min_next:
                continue

            k_max = k_min_next
            i_max = trajectories_info[seed_sequence.index([next_node])]['i_min']

        show_single_trajectory(fitting_info, candidates, node, k_max=k_max, i_max=i_max, ax=ax, trajectory_color=colors[i%len(colors)], **kwargs)

    if frame is not None:
        ax.imshow(frame, zorder=-100)
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)
        if heatmap is not None:
            ax.imshow(cv2.resize(heatmap, (frame.shape[1], frame.shape[0])), cmap='gray', vmin=0, vmax=1, zorder=-99, alpha=0.7, dpi=dpi)

    return ax


def show_neighboring_trajectories(frame_idx,
                                  fitting_info,
                                  candidates,
                                  path_mapping,
                                  frame,
                                  heatmap=None,
                                  num_prev=2,
                                  num_next=3,
                                  display='params k_min k_max',
                                  display_prev=None,
                                  display_next=None,
                                  color='w',
                                  color_prev='y',
                                  color_next='g',
                                  alpha=1,
                                  alpha_prev=0.6,
                                  alpha_next=0.6,
                                  ax=None,
                                  **kwargs):
    # show heatmap and detection candidates
    if heatmap is not None:
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        frame = cv2.addWeighted(frame, 0.5, cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB), 0.5, 0)

        # put red dot on the frame
        starting_frame = fitting_info['trajectories'][0]['k_seed']

        positions = candidates[frame_idx - starting_frame][:,::-1]
        positions = positions[np.where(positions[0]>=0)]
        frame = annotate_frame(frame, positions)

    node = path_mapping[frame_idx]

    prev_nodes = find_prev_nodes(frame_idx, path_mapping, num_prev)
    next_nodes = find_next_nodes(frame_idx, path_mapping, num_next)

    for prev_node in prev_nodes:
        ax = show_single_trajectory(fitting_info,
                                    candidates,
                                    prev_node,
                                    display=display_prev,
                                    alpha=alpha_prev,
                                    trajectory_color=color_prev,
                                    ax=ax,
                                    verbose=False,
                                    **kwargs)
    for next_node in next_nodes:
        ax = show_single_trajectory(fitting_info,
                                    candidates,
                                    next_node,
                                    display=display_next,
                                    alpha=alpha_next,
                                    trajectory_color=color_next,
                                    ax=ax,
                                    verbose=False,
                                    **kwargs)

    ax = show_single_trajectory(fitting_info,
                                candidates,
                                node,
                                display=display,
                                frame=frame,
                                stat_frame=frame_idx,
                                alpha=alpha,
                                trajectory_color=color,
                                ax=ax,
                                verbose=False,
                                **kwargs)

    im = figure_to_array(ax.figure)

    im2 = np.zeros(im.shape, dtype=im.dtype)
    s = -1
    im2[:-s] = im[s:]
    im2[-s:] = im[:s]
    im2[:1,] = 0

    return im2


def create_trajectory_video(train_configuration, filename=None, training_phase=None, show_heatmaps=True, split='val_1', dpi=100, num_frames=None, starting_frame=None, line_style='-', fitting_kw={}, **kwargs):
    """Create trajectory video. If num_frames is 0 or 1, an image will be created."""
    sf, candidates, n_candidates, values = get_candidates(train_configuration, training_phase, split)

    fitting_info = fit_trajectories(candidates, n_candidates, sf, **fitting_kw)
    trajectory_graph = build_trajectory_graph(fitting_info)
    shortest_paths = find_shortest_paths(trajectory_graph)
    path_mapping = build_path_mapping(fitting_info, shortest_paths)

    print("Rendering video")
    filename_src = get_video_source(train_configuration, split)

    # set fps and image size equal to source video
    cap = cv2.VideoCapture(filename_src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first_frame = cap.read()
    if ret:
        h, w = first_frame.shape[0], first_frame.shape[1]
    else:
        w, h = 1280, 720
    cap.release()

    if num_frames is None:
        num_frames = len(candidates)

    output_video = num_frames > 1
    if output_video:
        if filename is None:
            raise ValueError("Provide a filename for the video")
        out = cv2.VideoWriter(filename=filename,
                              fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                              fps=fps,
                              frameSize=(w, h))
    else:
        num_frames = 2

    # get starting frame
    if starting_frame is None:
        starting_frame = sf
    if starting_frame < sf:
        print(f"No fit for frames before {sf}, starting from {sf}")
        starting_frame = sf

    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    for i, frame in enumerate(frame_generator(filename_src, starting_frame+1, starting_frame+num_frames)):
        ax.cla()
        if i%100 == 0:
            gc.collect()
        heatmap=None
        if show_heatmaps:
            heatmap = get_heatmap(train_configuration, i+starting_frame+1, split, training_phase=training_phase)
        im2 = show_neighboring_trajectories(i+starting_frame,
                                            fitting_info,
                                            candidates,
                                            path_mapping,
                                            frame,
                                            heatmap,
                                            ax=ax,
                                            line_style=line_style,
                                            linewidth=1.5,
                                            **kwargs)

        if output_video:
            out.write(cv2.cvtColor(im2, cv2.COLOR_RGB2BGR))
    if output_video:
        out.release()
        plt.close(fig)
    else:
        ax.cla()
        ax.imshow(im2)
        if filename is not None:
            fig.savefig(filename)
        return fig, ax

    print("Done.")
