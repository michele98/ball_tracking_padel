import numpy as np
import matplotlib.pyplot as plt


def show_fit(background, trajectory, candidates, k_seed, k_min, k_mid, k_max, i_seed, i_min, i_mid, i_max):
    plt.imshow(background, zorder=-2)

    # trajectory
    k = np.arange(len(trajectory)) + k_seed - (len(trajectory)-1)//2
    plt.plot(trajectory[k<=k_min,1],trajectory[k<=k_min,0], 'y.-', zorder=-1, alpha=0.2)
    plt.plot(trajectory[k>=k_max,1],trajectory[k>=k_max,0], 'y.-', zorder=-1, alpha=0.2)
    mask = np.logical_and(k>=k_min, k<=k_max)
    plt.plot(trajectory[mask,1],trajectory[mask,0], 'y.-', zorder=-1, alpha=0.8)

    # support and seed
    plt.scatter(candidates[k_min, i_min, 1], candidates[k_min, i_min, 0], c='w', marker='^')
    plt.scatter(candidates[k_mid, i_mid, 1], candidates[k_mid, i_mid, 0], c='w')
    plt.scatter(candidates[k_max, i_max, 1], candidates[k_max, i_max, 0], c='w', marker='s')
    plt.scatter(candidates[k_seed, i_seed, 1], candidates[k_seed, i_seed, 0], c='k', s=5)

    plt.xlim(0, 640)
    plt.ylim(360, 0)
    plt.show()
