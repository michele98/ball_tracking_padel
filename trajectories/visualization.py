import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def visualize_trajectory_graph(trajectory_graph: nx.DiGraph, first_node: int = None, last_node: int = None, ax:plt.Axes=None):
    nodes_list = sorted(list(trajectory_graph.nodes()))

    if first_node is None:
        first_node_index = np.random.randint(len(nodes_list) - 20)
        first_node = nodes_list[first_node_index]
    else:
        nodes_before = [n for n in nodes_list if n<first_node]
        if len(nodes_before) > 0:
            first_node = nodes_before[-1]
        else:
            nodes_after = [n for n in nodes_list if n>=first_node]
            first_node = nodes_after[0]

    if last_node is None:
        last_node = nodes_list[nodes_list.index(first_node) + 20]

    G = trajectory_graph.subgraph(range(first_node, last_node))

    if ax is None:
        w, h, dpi = 1800, 600, 100
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    else:
        fig = ax.get_figure()

    # Create a linear layout
    pos = {node: (i, 0) for i, node in enumerate(sorted(G.nodes()))}

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


def show_fit(trajectory, candidates, k_seed, k_min, k_mid, k_max, i_seed, i_min, i_mid, i_max,
             background=None,
             ax=None,
             show_outside_range=True,
             show_fitting_points=True,
             annotate=False,
             trajectory_color='y'):

    if ax is None:
        w, h, dpi = 360*2, 360, 100
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)

    if background is not None:
        ax.imshow(background, zorder=-2)

    # trajectory
    k = np.arange(len(trajectory)) + k_seed - (len(trajectory)-1)//2
    if show_outside_range:
        ax.plot(trajectory[k<=k_min,1],trajectory[k<=k_min,0], f'{trajectory_color}.-', zorder=-1, alpha=0.2)
        ax.plot(trajectory[k>=k_max,1],trajectory[k>=k_max,0], f'{trajectory_color}.-', zorder=-1, alpha=0.2)
    mask = np.logical_and(k>=k_min, k<=k_max)
    ax.plot(trajectory[mask,1],trajectory[mask,0], f'{trajectory_color}.-', zorder=-1, alpha=0.8)

    # support and seed
    if show_fitting_points:
        ax.scatter(candidates[k_min, i_min, 1], candidates[k_min, i_min, 0], c='w', marker='^')
        ax.scatter(candidates[k_mid, i_mid, 1], candidates[k_mid, i_mid, 0], c='w')
        ax.scatter(candidates[k_max, i_max, 1], candidates[k_max, i_max, 0], c='w', marker='s')
        ax.scatter(candidates[k_seed, i_seed, 1], candidates[k_seed, i_seed, 0], c='k', s=5)

    if annotate:
        bbox = {'boxstyle': 'round',
                'facecolor': trajectory_color,
                'edgecolor': 'none',
                'alpha': 0.4}

        from matplotlib.font_manager import FontProperties
        font = FontProperties(family='sans-serif', weight='bold', size=12)

        ax.annotate(k_seed, [candidates[k_seed, i_seed, 1], candidates[k_seed, i_seed, 0]], fontproperties=font, bbox=bbox, color='k')
        if show_fitting_points:
            ax.annotate(k_min, [candidates[k_min, i_min, 1], candidates[k_min, i_min, 0]], fontproperties=font, bbox=bbox, color='w')
            ax.annotate(k_mid, [candidates[k_mid, i_mid, 1], candidates[k_mid, i_mid, 0]], fontproperties=font, bbox=bbox, color='w')
            ax.annotate(k_max, [candidates[k_max, i_max, 1], candidates[k_max, i_max, 0]], fontproperties=font, bbox=bbox, color='w')

    return ax
