import numpy as np
from scipy import ndimage

def anisotropic_diffusion(image, niter=1, kappa=50, gamma=0.1, step=(1., 1.), option=1):
    """
    Anisotropic diffusion.

    image - input image
    niter - number of iterations
    kappa - conduction coefficient
    gamma - max value of 1/sqrt(2) of the edge detection operator
    step - tuple, the distance between adjacent pixels in y and x directions
    option - 1 Perona-Malik, 2 Tukey biweight

    Return the diffused image.
    """
    # Convertion image en float32
    image = np.array(image, dtype=np.float32)

    # Define conduction gradients functions
    if option == 1:
        diff_op = lambda x: np.exp(-(x/kappa)**2)
    elif option == 2:
        diff_op = lambda x: 1./(1.+(x/kappa)**2)

    # Initialisation image de sortie
    diff = np.zeros_like(image)

    # Initialisation des variables
    deltaS = np.zeros_like(image)
    deltaE = np.zeros_like(image)
    gS = np.zeros_like(image)
    gE = np.zeros_like(image)

    for ii in range(niter):
        # Calcul des gradients
        gS[:-1,:] = np.diff(diff, axis=0)
        gE[:,:-1] = np.diff(diff, axis=1)

        # Calculate the diffusion coefficients
        if option == 1:
            deltaS[:-1,:] = diff_op(gS[:-1,:])
            deltaE[:,:-1] = diff_op(gE[:,:-1])
        elif option == 2:
            deltaS[:-1,:] = diff_op(gS[:-1,:])**2
            deltaE[:,:-1] = diff_op(gE[:,:-1])**2

        # Mise à jour de l'image 
        diff[1:,:] -= deltaS[1:,:]*step[0]
        diff[:-1,:] += deltaS[:-1,:]*step[0]
        diff[:,1:] -= deltaE[:,1:]*step[1]
        diff[:,:-1] += deltaE[:,:-1]*step[1]

    return diff



from scipy import misc

# Charge l'image
image = misc.ascent()

# Applique diffusion
diff = anisotropic_diffusion(image, niter=50, kappa=10)

# Affiche res
import matplotlib.pyplot as plt
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)

ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title('Original')
ax0.axis('off')
ax1.imshow(diff, cmap=plt.cm.gray)
ax1.set_title('Perona Malik')
ax1.axis('off')
plt.show()
