import io
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from trajectories.data_reading import get_frame


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
                           k_min = None,
                           i_min = None,
                           k_max = None,
                           i_max = None,
                           ax=None,
                           show_outside_range=False,
                           display='k_min',
                           annotate=True,
                           show_fitting_points=False,
                           trajectory_color='y',
                           frame=None,
                           fontsize=12):
    if ax is None:
        w, h, dpi = 1280, 720, 100
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)

    trajectories_info = fitting_info['trajectories']
    starting_frame = trajectories_info[0]['k_seed']

    k_seed_sequence = [t['k_seed'] for t in trajectories_info]
    trajectory_info = trajectories_info[k_seed_sequence.index(k_seed)]

    if not trajectory_info['found_trajectory']:
        print(f'No fitted trajectory for frame {k_seed}')
        return ax

    trajectory = trajectory_info['trajectory']

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

    k_seed -= starting_frame
    i_seed = trajectory_info['i_seed']

    # trajectory
    k = np.arange(len(trajectory)) + k_seed - (len(trajectory)-1)//2
    if show_outside_range:
        ax.plot(trajectory[k<=k_min,1],trajectory[k<=k_min,0], f'{trajectory_color}.-', zorder=-1, alpha=0.2)
        ax.plot(trajectory[k>=k_max,1],trajectory[k>=k_max,0], f'{trajectory_color}.-', zorder=-1, alpha=0.2)
    mask = np.logical_and(k>=k_min, k<=k_max)
    ax.plot(trajectory[mask,1],trajectory[mask,0], f'{trajectory_color}.-', zorder=-1, alpha=0.8)

    if display is not None:
        bbox = {'boxstyle': 'round',
                'facecolor': trajectory_color,
                'edgecolor': 'none',
                'alpha': 0.4}

        if 'all' in display:
            display = ['k_min', 'k_mid', 'k_max', 'k_seed']

        font = FontProperties(family='sans-serif', weight='bold', size=fontsize)

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

    if frame is not None:
        ax.imshow(frame, zorder=-2)
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)

    ax.set_axis_off()
    return ax


def show_trajectory_path(fitting_info: dict,
                         candidates: np.ndarray,
                         path: list,
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
    path : list of int
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
    for i, node in enumerate(path):
        k_max = None
        i_max = None
        if link_trajectories and i < len(path)-1:
            next_node = path[i+1]

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
            ax.imshow(cv2.resize(heatmap, (frame.shape[1], frame.shape[0])), cmap='gray', vmin=0, vmax=1, zorder=-99, alpha=0.7)

    fig.tight_layout(pad=0)
    return ax


def figure_to_array(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im[:,:,:3]
