import numpy as np
from numba import jit


@jit
def euclidean_distance(pos1, pos2):
    delta = pos1 - pos2
    return np.sqrt(delta[0]**2 + delta[1]**2)


@jit
def find_seed_triplets(candidates: np.ndarray, n_candidates: np.ndarray, k: int, radius=100):
    """Find seed triplets for the given candidate list

    Parameters
    ----------
    candidates : np.ndarray, shape (:, max_candidates, 2)
        positions of the detection candidates.
        The first dimension refers to the frames,
        the second dimension to the candidate in each frame
        and the third one to the x and y components: the first element is y, the second one x.
    n_candidates : 1D np.ndarray
        number of candidates in each frame. Necessary for jit complation
    k : int
        frame of which to find the seed triplet
    radius : int, optional
        maximum distance between candidates of different frames
        to use them for a seed triplet, by default 100

    Returns
    -------
    seed_triplets: np.ndarray of shape (num_triplets, 3).
        The second component contains the indices of the candidates in:
         - k-1
         - k
         - k+1
        respectively.
    """
    seed_triplets_i = []

    for i, candidate in enumerate(candidates[k, :n_candidates[k]]):
        for i_prev, prev_candidate in enumerate(candidates[k-1, :n_candidates[k-1]]):
            for i_next, next_candidate in enumerate(candidates[k+1, :n_candidates[k+1]]):
                if euclidean_distance(candidate, prev_candidate) < radius and euclidean_distance(candidate, next_candidate) < radius:
                    seed_triplets_i.append([i_prev, i, i_next])
    return np.asarray(seed_triplets_i)


@jit
def estimate_parameters(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, dk21: int, dk32: int):
    """Estimate parameters for a constant acceleration model (a parabola).
        The fitted model is:
         - a = 2 * (dk21 * (p3-p2) - dk32 * (p2-p1)) / (dk21 * dk32 * (dk21+dk32))
         - v1 = (p2-p1)/dk21 - dk21*a/2

    Parameters
    ----------
    p1 : np.ndarray
        position vector of the first point
    p2 : np.ndarray
        position vector of the second point
    p3 : np.ndarray
        position vector of the third point
    dk21 : int
        difference between the timesteps of the first and second points
    dk32 : int
        difference between the timesteps of the second and third points

    Returns
    -------
    v1, a: np.ndarray
        v1: velocity vector between the first and second point
        a: acceleration y
    """
    a = 2 * (dk21 * (p3-p2) - dk32 * (p2-p1)) / (dk21 * dk32 * (dk21+dk32))
    v1 = (p2-p1)/dk21 - dk21*a/2
    return v1, a


@jit
def estimate_position(pos, v, a, dk):
    return pos + dk*v + dk*dk*a/2


@jit
def compute_trajectory(seed_position: np.ndarray, v: np.ndarray, a: np.ndarray, window_size: int, window_center: int = 0):
    """Compute the estimated positions along each timestep in the window

    Parameters
    ----------
    seed_position : np.ndarray
        starting position
    v : np.ndarray
        velocity vector between the first and second point 
    a : np.ndarray
        acceleration vector
    window_size : int
        size of the window
    window_center : int
        center of the window. If 0, the window is centered on the seed position

    Returns
    -------
    np.ndarray of shape(window_size, 2)
        contains all the estimated positions
    """

    k0 = window_center - (window_size-1)//2

    positions = np.zeros((window_size, 2))
    for k in range(len(positions)):
        positions[k] = estimate_position(seed_position, v, a, k+k0)
    return positions


@jit
def find_next_triplet(trajectory: np.ndarray, candidates: np.ndarray, n_candidates: np.ndarray, d_threshold: float, window_size: int, window_center: int):
    """Find the next triplet of

    Parameters
    ----------
    trajectory : np.ndarray
        trajectory of the ball
    candidates : np.ndarray, shape (:, max_candidates, 2)
        positions of the detection candidates.
        The first dimension refers to the frames,
        the second dimension to the candidate in each frame
        and the third one to the x and y components: the first element is y, the second one x.
    n_candidates : 1D np.ndarray
        number of candidates in each frame. Necessary for jit complation
    d_threshold : float
        maximum distance between the true position of the candidates and the estimated position
    window_size : int
    window_center : int
        center of the window in frames. The center must be the seed frame.

    Returns
    -------
    k_min, k_mid, k_max: int
        frame indices of the triplet
    i_min, i_mid, i_max: int
        candidate indices in their frame
    """
    support_k = []
    support_i = []

    k0 = window_center - (window_size-1)//2

    for k in range(window_size):
        estimated_position = trajectory[k]
        for i in range(n_candidates[k+k0]):
            d = euclidean_distance(candidates[k+k0,i], estimated_position)
            if d<d_threshold:
                support_k.append(k+k0)
                support_i.append(i)

    k_min, k_mid, k_max = 0, 0, 0
    i_min, i_mid, i_max = 0, 0, 0

    if len(support_k) >= 3:
        k_min = support_k[0]
        i_min = support_i[0]

        k_max = support_k[-1]
        i_max = support_i[-1]

        support_k = np.asarray(support_k)
        s_mid = np.argmin(np.abs(np.abs(k_max-support_k) - np.abs(k_min-support_k)))

        k_mid = support_k[s_mid]
        i_mid = support_i[s_mid]

    return k_min, k_mid, k_max, i_min, i_mid, i_max


# TODO: implement trajectory cost
# @jit
# def trajectory_cost(trajectory, candidates, n_candidates, window_size, window_center):
#     for k in range(len(candidates)):
#         for j in range(len(candidates[k])):

#     return np.sum((true_trajectory - predicted_trajectory)**2)
