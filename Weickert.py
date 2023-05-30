import cv2
import numpy as np

def g(n):
    return 1 / (1 + n)

def weickert_diffusivity(dux, duy, epsilon=0.001):
    norm_grad = np.sqrt(np.square(dux) + np.square(duy))
    c = 1 / np.sqrt(epsilon**2 + norm_grad**2)
    return c

def calc_divergence(u, g, epsilon=0.001):
    dux, duy = np.gradient(u, edge_order=2)

    diffusivity = weickert_diffusivity(dux, duy, epsilon)

    div_x = g(np.abs(dux)) * diffusivity * dux
    div_y = g(np.abs(duy)) * diffusivity * duy

    divergence = np.zeros_like(u)
    divergence[:, :-1] += div_x[:, :-1]
    divergence[:, 1:] -= div_x[:, :-1]
    divergence[:-1, :] += div_y[:-1, :]
    divergence[1:, :] -= div_y[:-1, :]

    return divergence

def anisotropic_diffusion_weickert(u, n_iter, dt, epsilon=0.001):
    for _ in range(n_iter):
        divergence = calc_divergence(u, g, epsilon)
        u += dt * u * np.power(divergence, 2)

    return u

# Charger l'image en niveaux de gris
img = cv2.imread('paysage.jpg', cv2.IMREAD_GRAYSCALE)

# Normaliser l'image
img_normalized = img.astype(np.float32) / 255.0

# Appliquer la diffusion anisotrope avec Weickert
n_iter = 10
dt = 0.1
epsilon = 0.001
result = anisotropic_diffusion_weickert(img_normalized, n_iter, dt, epsilon)

# Reconvertir l'image normalisée en niveaux de gris
result_scaled = (result * 255).astype(np.uint8)

scale_percent = 10 # percent of original size
width = int(result_scaled.shape[1] * scale_percent / 100)
height = int(result_scaled.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(result_scaled, dim, interpolation = cv2.INTER_AREA)

# Afficher l'image résultante
cv2.imshow('Diffusion anisotrope avec Weickert', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
