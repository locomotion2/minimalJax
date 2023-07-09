import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from moviepy.editor import ImageSequenceClip
from functools import partial
import proglog
import importlib

import warnings


def make_plot(ax, i, cart_coords, l1, l2, max_trail=30, trail_segments=20,
              r=0.05):
    # Plot and save an image of the double pendulum configuration for time step i.
    plt.cla()

    x1, y1, x2, y2 = cart_coords
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')  # rods
    c0 = Circle((0, 0), r / 2, fc='k', zorder=10)  # anchor point
    c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)  # mass 1
    c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)  # mass 2
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # plot the pendulum trail (ns = number of segments)
    s = max_trail // trail_segments
    for j in range(trail_segments):
        imin = i - (trail_segments - j) * s
        if imin < 0: continue
        imax = imin + s + 1
        alpha = (j / trail_segments) ** 2  # fade the trail into alpha
        ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
                lw=2, alpha=alpha)

    # Center the image on the fixed anchor point. Make axes equal.
    ax.set_xlim(-l1 - l2 - r, l1 + l2 + r)
    ax.set_ylim(-l1 - l2 - r, l1 + l2 + r)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    # plt.savefig('./frames/_img{:04d}.png'.format(i//di), dpi=72)


def radial2cartesian(t1, t2, l1, l2):
    # Convert from radial to Cartesian coordinates.
    x1 = l1 * np.cos(t1)
    y1 = l1 * np.sin(t1)
    x2 = x1 + l2 * np.cos(t1 + t2)
    y2 = y1 + l2 * np.sin(t1 + t2)
    return x1, y1, x2, y2


def fig2image(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def save_movie(images, path, duration=100, loop=0, **kwargs):
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=duration, loop=loop, **kwargs)

def make_images(x_traj, N, di, L1, L2):
    theta1, theta2 = x_traj[:, 0], x_traj[:, 1]
    cart_coords = radial2cartesian(theta1, theta2, L1, L2)

    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)
    warnings.filterwarnings("ignore")

    images = []
    N_images = N - 1
    for i in range(0, N_images, di):
        print("{}/{}".format(i // di, N_images // di),
              end='\n' if i // di % 20 == 0 else ' ')
        make_plot(ax, i, cart_coords, L1, L2)
        images.append(fig2image(fig))

    # importlib.reload(proglog)
    # proglog.default_bar_logger = partial(proglog.default_bar_logger, None)
    # ImageSequenceClip(images, fps=25).ipython_display()

    return images

