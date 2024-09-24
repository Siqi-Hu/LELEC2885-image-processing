import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

def plot_pyramid(arrays):
    H, W = arrays[0].shape
    next_shapes = [x.shape for x in arrays[1:]]+[(0,0)]
    W = W + next_shapes[0][1]
    image = np.zeros((H, W))+np.nan
    y, x = 0, 0
    for i, (array, next_shape) in enumerate(zip(arrays, next_shapes)):
        h, w = array.shape
        nh, nw = next_shape
        image[y:y+h,x:x+w] = array
        if i % 4 == 0:    # To the right
            y, x = y, x+w
        elif i % 4 == 1:  # To the bottom
            y, x = y+h, x+nw
        elif i % 4 == 2:  # To the left
            y, x = y+nh, x-nw
        elif i % 4 == 3:  # to the top
            y, x = y-nh, x
    plt.imshow(image, cmap="gray")
    
    
# Function to display each pixels in an image with grid
def display_pixels(ax, image, title=None, **kwargs):
    # Set colormap to 'gray'
    mpl.rc('image', cmap='gray')
    # Set np.nan to red
    mpl.cm.get_cmap().set_bad(color='red')

    height, width = image.shape[0:2]
    ax.imshow(image, **kwargs)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    # if image is small enough, draw pixel borders
    if width < 100 and height < 100:
        ax.set_xticks(np.arange(-.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-.5, height, 1), minor=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(which='minor', color=(.5,.5,.5), linestyle='-', linewidth=1)

    for tic in ax.xaxis.get_minor_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax.yaxis.get_minor_ticks():
        tic.tick1On = tic.tick2On = False
    if title:
        ax.set_title(title)
