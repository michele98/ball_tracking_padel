import numpy as np
import matplotlib.pyplot as plt


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
