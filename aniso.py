import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi


def anisotropic_diffusion():
    mpl.rcParams['font.size'] = 10

    # Generate noisy image of a square
    im = np.zeros((256, 256))
    im[64:-64, 64:-64] = 1

    im = ndi.rotate(im, 15, mode='constant')
    im = ndi.gaussian_filter(im, 8)
    im += 0.2 * np.random.random(im.shape)

    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(im)
    edges2 = feature.canny(im, sigma=3)

    # display results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)

    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)

    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

    fig.tight_layout()

    plt.show()