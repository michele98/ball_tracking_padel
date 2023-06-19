import numpy as np
from numba import jit

import networkx as nx
from networkx import DiGraph, dijkstra_path

"""All this is based on this paper: https://www.researchgate.net/publication/4246136_A_Novel_Data_Association_Algorithm_for_Object_Tracking_in_Clutter_with_Application_to_Tennis_Video_Analysis"""


@jit
def euclidean_distance(pos1, pos2, axis=0):
    delta = pos1 - pos2
    if axis==1:
        delta = delta.T
    return np.sqrt(delta[0]**2 + delta[1]**2)


@jit
def trajectory_distance(trajectory_1: np.ndarray, support_1: np.ndarray, k_seed_1: int, trajectory_2: np.ndarray, support_2: np.ndarray, k_seed_2: int):
    """Calculate the distance between 2 trajectories.
    The trajectories are the array of the estimated positions

    trajectory_1 : np.ndarray of shape (window_size,)
        first trajectory
    support_1 : np.ndarray of shape (:, 2)
        supports of first trajectory
    k_seed_1 : int
        seed frame of the first trajectory
    trajectory_2 : np.ndarray of shape (window_size,)
        second trajectory
    support_2 : np.ndarray of shape (:, 2)
        supports of second trajectory
    k_seed_2 : int
        seed frame of the second trajectory

    Returns
    -------
    distance : float
        distance between the two trajectories
    """

    if len(np.intersect1d(np.arange(support_1[0,0], support_1[-1,0]), np.arange(support_2[0,0], support_2[-1,0]))) > 0:
        # trajectories are overlapping
        distance = 0
        for k in range(max(support_1[0,0], support_2[0,0]), min(support_1[-1,0], support_2[-1,0])):
            k1_index = np.where(support_1[:,0]==k)[0]
            k2_index = np.where(support_2[:,0]==k)[0]

            if (len(k1_index)==0 and len(k2_index)==0):
                pass
            elif (len(k1_index)==0 and len(k2_index)>0) or (len(k1_index)>0 and len(k2_index)==0):
                distance = np.inf
            elif support_1[k1_index[0], 1] != support_2[k2_index[0], 1]:
                distance = np.inf
    else:
        if k_seed_2 > k_seed_1:
            dk = k_seed_2 - k_seed_1
            if dk >= len(trajectory_1):
                distance = np.inf
            else:
                distance = np.min(euclidean_distance(trajectory_1[dk], trajectory_2[:len(trajectory_1)-dk], axis=1))
        else:
            dk = k_seed_1 - k_seed_2
            if dk >= len(trajectory_1):
                distance = np.inf
            else:
                distance = np.min(euclidean_distance(trajectory_2[dk], trajectory_1[:len(trajectory_1)-dk], axis=1))

    return distance


def build_trajectory_graph(fitting_info: dict):
    """Build trajectory graph for the found trajectories

    Parameters
    ----------
    fitting_info : dict
        contains the fitting info

    Returns
    -------
    trajectory_graph: networkx.DiGraph
        weighted directed graph linking the trajectories
    """
    print("Building trajectory graph:")
    trajectory_graph = DiGraph()

    N = fitting_info['parameters']['N']
    trajectory_info = fitting_info['trajectories']

    for i in range(len(trajectory_info)):
        print(f"{i+1} of {len(trajectory_info)}", end='\r')
        if not trajectory_info[i]['found_trajectory']:
            continue
        for j in range(i, min(i+N, len(trajectory_info))):
            if not trajectory_info[j]['found_trajectory']:
                continue
            t1 = trajectory_info[i]['trajectory']
            s1 = trajectory_info[i]['support']
            k1 = trajectory_info[i]['k_seed']

            t2 = trajectory_info[j]['trajectory']
            s2 = trajectory_info[j]['support']
            k2 = trajectory_info[j]['k_seed']

            d = trajectory_distance(t1, s1, k1, t2, s2, k2)

            if d != np.inf and i!=j:
                trajectory_graph.add_edge(k1, k2, weight=d)

    print(f"{i+1} of {len(trajectory_info)}")
    print("Done.")
    return trajectory_graph


def find_shortest_paths(trajectory_graph: nx.DiGraph):
    wcc = list(nx.weakly_connected_components(trajectory_graph))
    paths = []
    for el in wcc:
        try:
            paths.append(dijkstra_path(trajectory_graph, min(el), max(el)))
        except nx.NetworkXNoPath as e:
            pass
    return paths


def build_path_mapping(fitting_info: dict, shortest_paths: list):
    """Build mapping from each frame to the corresponding fitted path

    Parameters
    ----------
    fitting_info : dict
        _description_
    shortest_paths : list
        shortest path in the trajectory graph

    Returns
    -------
    mapping : dict
        mapping from frame to corresponding node in the graph.
        It associates a trajectory to each frame.

        If no trajectory is found, the value is None.
    """
    trajectories_info = fitting_info['trajectories']

    frame_sequence = [t['k_seed'] for t in trajectories_info]
    path = [p for path in shortest_paths for p in path]

    first_frames = [trajectories_info[frame_sequence.index(node)]['k_min'] for node in path]
    last_frames = [trajectories_info[frame_sequence.index(node)]['k_max'] for node in path]

    mapping = {}
    for frame_idx in range(trajectories_info[0]['k_seed'], trajectories_info[-1]['k_seed']):
        mapping[frame_idx] = None
        for k_min, k_seed, k_max in zip(first_frames, path, last_frames):
            if k_min <= frame_idx and k_max >= frame_idx:
                mapping[frame_idx] = k_seed
                break
    return mapping


def find_next_node(frame_idx, path_mapping):
    node = path_mapping[frame_idx]
    next_node = None
    next_idx = None
    for f, n in path_mapping.items():
        if f <= frame_idx:
            continue
        if n!=node:
            next_node = n
            next_idx = f
            break
    return next_idx, next_node


def find_next_nodes(frame_idx, path_mapping, n=1):
    if n==0:
        return []
    next_nodes = []
    idx = frame_idx
    for _ in range(n):
        idx, next_node = find_next_node(idx, path_mapping)
        # if next_node is None:
        #     break
        next_nodes.append(next_node)

    return next_nodes


def find_prev_node(frame_idx, path_mapping):
    node = path_mapping[frame_idx]
    prev_node = None
    prev_idx = None
    for f, n in path_mapping.items():
        if n!=node:
            prev_node = n
            prev_idx = f
            continue
        if f >= frame_idx:
            break
    return prev_idx, prev_node


def find_prev_nodes(frame_idx, path_mapping, n=1):
    if n==0:
        return []
    previous_nodes = []
    idx = frame_idx
    for _ in range(n):
        idx, previous_node = find_prev_node(idx, path_mapping)
        # if previous_node is None:
        #     break
        previous_nodes.append(previous_node)

    return list(reversed(previous_nodes))