import numpy as np
from numba import jit

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
