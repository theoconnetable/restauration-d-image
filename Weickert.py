import cv2
import numpy as np

def weickert_diffusion(image, num_iterations, kappa):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialiser les matrices de diffusion
    Ix2 = np.zeros_like(gray, dtype=np.float32)
    Iy2 = np.zeros_like(gray, dtype=np.float32)
    Ixy = np.zeros_like(gray, dtype=np.float32)

    # Calculer les dérivées de l'image
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Calculer les composantes du tenseur de diffusion
    Ix2 = sobelx * sobelx
    Iy2 = sobely * sobely
    Ixy = sobelx * sobely

    # Appliquer la gaussienne au tenseur de diffusion
    Ix2 = cv2.GaussianBlur(Ix2, (3, 3), 0)
    Iy2 = cv2.GaussianBlur(Iy2, (3, 3), 0)
    Ixy = cv2.GaussianBlur(Ixy, (3, 3), 0)

    for _ in range(num_iterations):
        # Calculer les valeurs propres du tenseur de diffusion
        lambda1 = 0.5 * (Ix2 + Iy2 + np.sqrt((Ix2 - Iy2) ** 2 + 4 * Ixy * Ixy))
        lambda2 = 0.5 * (Ix2 + Iy2 - np.sqrt((Ix2 - Iy2) ** 2 + 4 * Ixy * Ixy))

        # Calculer les coefficients de diffusion
        c1 = np.exp(-((lambda1 - kappa) ** 2) / (2 * kappa ** 2))
        c2 = np.exp(-((lambda2 - kappa) ** 2) / (2 * kappa ** 2))

        # Diffuser l'image selon les valeurs propres du tenseur de diffusion
        diffused = c1 * sobelx + c2 * sobely

        # Mettre à jour les dérivées de l'image avec la diffusion
        sobelx = diffused * sobelx
        sobely = diffused * sobely

    # Reconstruire l'image couleur en utilisant les dérivées diffusées
    filtered = cv2.merge((sobelx.astype(np.uint8), sobely.astype(np.uint8), sobely.astype(np.uint8)))


    # Ajouter l'image originale et l'image filtrée pour obtenir le résultat final
    result = cv2.addWeighted(image, 0.5, filtered, 0.5, 0)

    return result


# Charger l'image
image = cv2.imread("paysage.jpg")

# Paramètres de la diffusion anisotropique
num_iterations = 5  # Nombre d'itérations de diffusion
kappa = 0.2  # Paramètre de régularisation

# Appliquer la diffusion anisotropique
result = weickert_diffusion(image, num_iterations, kappa)

scale_percent = 10 # percent of original size
width = int(result.shape[1] * scale_percent / 100)
height = int(result.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
res = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)
img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

concat  = np.concatenate((img, res), axis=1)

# Afficher l'image originale et le résultat
cv2.imshow("Diffusion par Weickert", concat)
cv2.waitKey(0)
cv2.destroyAllWindows()
