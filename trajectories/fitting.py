import numpy as np
from numba import jit

from trajectories.utils import *

"""All this is based on this paper: https://www.researchgate.net/publication/4246136_A_Novel_Data_Association_Algorithm_for_Object_Tracking_in_Clutter_with_Application_to_Tennis_Video_Analysis"""


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
    seed_triplets: np.ndarray of shape (num_seed_triplets, 3).
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
                if squared_distance(candidate, prev_candidate) < radius**2 and squared_distance(candidate, next_candidate) < radius**2:
                    seed_triplets_i.append([i_prev, i, i_next])

    if len(seed_triplets_i) == 0:
        return None
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
def find_support(trajectory: np.ndarray, candidates: np.ndarray, n_candidates: np.ndarray, d_threshold: float, window_size: int, window_center: int):
    """Find the support of the given trajectory

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
        in the previous iteration
    window_size : int
    window_center : int
        center of the window in frames. The center must be the seed frame.

    Returns
    -------
    support: np.ndarray of shape (support_size, 2)
        frame indices in (:,0) and candidate indices in (:,1)
    """
    support_k = []
    support_i = []

    k0 = window_center - (window_size-1)//2

    for k in range(window_size):
        estimated_position = trajectory[k]

        candidate_idx = -1
        d2_min = d_threshold**2
        for i in range(n_candidates[k+k0]):
            d2 = squared_distance(candidates[k+k0,i], estimated_position)
            if d2<d2_min:
                candidate_idx = i
                d2_min = d2
        if candidate_idx >= 0:
            support_k.append(k+k0)
            support_i.append(candidate_idx)

    support = np.zeros((len(support_k), 2), dtype=np.int32)
    support[:, 0] = support_k
    support[:, 1] = support_i
    return support


@jit
def find_next_triplet(support: np.ndarray):
    """Find the next frame triplet from the given support.

    Parameters
    ----------
    support: np.ndarray of shape (support_size, 2)
        frame indices in (:,0) and candidate indices in (:,1)

    Returns
    -------
    k_min, k_mid, k_max: int
        frame indices of the triplet
    i_min, i_mid, i_max: int
        candidate indices in their frame
    """
    k_min, k_mid, k_max = -1, -1, -1
    i_min, i_mid, i_max = -1, -1, -1

    if len(support) >= 3:
        k_min = support[0,0]
        i_min = support[0,1]

        k_max = support[-1,0]
        i_max = support[-1,1]

        s_mid = find_middle_support_index(support)

        k_mid = support[s_mid,0]
        i_mid = support[s_mid,1]

    return k_min, k_mid, k_max, i_min, i_mid, i_max


@jit
def trajectory_cost(trajectory, candidates, n_candidates, d_threshold, window_size, window_center):
    k0 = window_center - (window_size-1)//2

    cost = 0
    for k, position in enumerate(trajectory):
        for i in range(n_candidates[k+k0]):
            d2 = squared_distance(position, candidates[k+k0,i])
            d2 = min(d2, d_threshold**2)
            cost += d2
    return cost


@jit
def fit_trajectories(candidates: np.ndarray, n_candidates: np.ndarray, k_seed: int, seed_radius: float, d_threshold: float, N: int):
    """Fit trajectories to position candidates.
    Seed triplets are found first, and then for each seed triplet a trajectory is iteratively fitted.

    Parameters
    ----------
    candidates : np.ndarray, shape (:, max_candidates, 2)
        positions of the detection candidates.
        The first dimension refers to the frames,
        the second dimension to the candidate in each frame
        and the third one to the x and y components: the first element is y, the second one x.
    n_candidates : 1D np.ndarray
        number of candidates in each frame. Necessary for jit complation
    k_seed : int
        seed frame from which to start calculating the ball trajectories.
        It is the central frame of each seed triplet.
    radius : int
        maximum distance between candidates of different frames
        to use them for a seed triplet, by default 100
    d_threshold : float
        maximum distance between the true position of the candidates and the estimated position
        in the previous iteration
    N : int
        number of frames before and after to use for the trajectory fitting.
        The window size will therefore be 2*N+1

    Returns
    -------
    parameters : np.ndarray of shape (num_seed_triplets, 2, 2)
        the parameters of the fitted parabolic trajectories for each seed triplet.
        In the second dimension the order is: v, a
    info : np.ndarray of shape (num_seed_triplets, 9)
        Information about the candidates in the support for each seed triplet.
        The values in the second component correspond respectively to:
         - `'k_seed'`: index of the seed frame
         - `'k_min'`: index of the first frame used to fit the trajectory
         - `'k_mid'`: index of the second frame used to fit the trajectory
         - `'k_max'`: index of the third frame used to fit the trajectory
         - `'i_seed'`: index of the candidate in the seed frame
         - `'i_min'`: index of the candidate in the first frame
         - `'i_mid'`: index of the candidate in the second frame
         - `'i_max'`: index of the candidate in the third frame
         - `'iterations'`: number of iterations before convergence

    trajectories : np.ndarray of shape (num_seed_triplets, 2*N+1, 2)
        fitted trajectories along the whole window
    costs : np.ndarray of shape (num_seed_triplets)
        costs of each trajectory. It is computed as in equation (7) of the paper
    """
    window_size = 2*N+1

    seed_triplets = find_seed_triplets(candidates, n_candidates, k_seed, radius=seed_radius)

    if seed_triplets is None:
        return None, None, None, None, None

    parameters = np.zeros((len(seed_triplets), 2, 2)) # v, a
    info = np.zeros((len(seed_triplets), 10), dtype=np.uint32)
    trajectories = np.zeros((len(seed_triplets), window_size, 2), dtype=np.float32) - 1
    supports = np.zeros((len(seed_triplets), window_size, 2), dtype=np.int32) -1
    costs = np.zeros(len(seed_triplets), dtype=np.float32) + np.finfo(np.float32).max

    for s, seed_triplet in enumerate(seed_triplets):
        k_min, k_mid, k_max = k_seed-1, k_seed, k_seed+1
        i_min, i_mid, i_max = seed_triplet
        i_seed = i_min

        cost_old = np.inf # initialize old cost to infinity
        support_old = np.array([[0]], dtype=np.int32)   # initialize old support to 0

        v, a = estimate_parameters(candidates[k_min, i_min], candidates[k_mid, i_mid], candidates[k_max, i_max], 1, 1)
        for i in range(N):
            trajectory = compute_trajectory(candidates[k_min, i_min], v, a, window_size, k_seed-k_min) # centered around k_seed, start computing from k_min

            support = find_support(trajectory, candidates, n_candidates, d_threshold, window_size, k_seed)
            cost = trajectory_cost(trajectory, candidates, n_candidates, d_threshold, window_size, k_seed)

            if k_max == -1 or len(support) <= len(support_old) or cost > cost_old:
                trajectories[s] = trajectory
                costs[s] = cost
                parameters[s,0] = v
                parameters[s,1] = a
                supports[s, :len(support)] = support
                for j, n in enumerate([k_seed, k_min, k_mid, k_max, i_seed, i_min, i_mid, i_max, len(support), i+1]):
                    info[s, j] = n
                break

            support_old = support
            cost_old = cost

            k_min, k_mid, k_max, i_min, i_mid, i_max = find_next_triplet(support)
            v, a = estimate_parameters(candidates[k_min, i_min], candidates[k_mid, i_mid], candidates[k_max, i_max], k_mid-k_min, k_max-k_mid)

    costs = np.where(np.isnan(costs), np.inf, costs) # turn nan costs to infinity
    return parameters, info, trajectories, supports, costs
