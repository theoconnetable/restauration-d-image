import numpy as np
import cv2

def g(x):
    """Fonction de tendance à diminuer sur l'intervalle [0, inf[."""
    return 1 / (1 + x**2)

##def gradient_x(img):
##    """Calcul de la composante x du gradient de l'image."""
##    img = np.array(img)
##    h, w = img.shape[:2]
##    gx = np.zeros((h, w))
##    for y in range(h):
##        gx[y, 0] = img[y, 1] - img[y, 0]
##        gx[y, w-1] = img[y, w-1] - img[y, w-2]
##    for x in range(1, w-1):
##        for y in range(h):
##            gx[y, x] = (img[y, x+1] - img[y, x-1]) / 2
##    return gx
##
##def gradient_y(img):
##    """Calcul de la composante y du gradient de l'image."""
##    h, w = img.shape[:2]
##    gy = np.zeros((h, w))
##    for x in range(w):
##        gy[0, x] = img[1, x] - img[0, x]
##        gy[h-1, x] = img[h-1, x] - img[h-2, x]
##    for x in range(w):
##        for y in range(1, h-1):
##            gy[y, x] = (img[y+1, x] - img[y-1, x]) / 2
##    return gy


def gradient_x(img):
    img = np.array(img)
    h, w = img.shape[:2]
    gx = np.zeros((h, w))
    for y in range(h):
        for x in range(1, w-1):
            gx[y,x] = (img[y,x+1] - img[y,x-1]) / 2
    return gx

def gradient_y(img):
    img = np.array(img)
    h, w = img.shape[:2]
    gy = np.zeros((h, w))
    for y in range(1,h-1):
        for x in range(w):
            gy[y,x] = (img[y+1,x] - img[y-1,x]) / 2
    return gy

def test(img):
    h, w = img.shape[:2]
    gx = gradient_x(img)
    gy = gradient_y(img)
    div = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            div[y,x] = g(abs(gx[y,x])) * gx[y,x] * ( x + 1/2) - g(abs(gx[y,x])) * gx[y,x] * ( x - 1/2)+ g(abs(gy[y,x])) * gy[y,x] * (y + 1/2) - g(abs(gy[y,x])) * gy[y,x] * (y - 1/2)
    return div


def divergence(img):
    """Calcul de la divergence du champ de vecteurs formé par le gradient de l'image."""
    h, w = img.shape[:2]
    gx = gradient_x(img)
    gy = gradient_y(img)
    div = np.zeros((h, w))
    ##print("VALEUR ENTREE" + str(img))
    for x in range(1, w-1):
        div[0, x] = -g(abs(gx[0, x])) * gx[0, x]
        div[h-1, x] = -g(abs(gx[h-1, x])) * gx[h-1, x]
    for y in range(1, h-1):
        div[y, 0] = -g(abs(gy[y, 0])) * gy[y, 0]
        div[y, w-1] = -g(abs(gy[y, w-1])) * gy[y, w-1]
    for x in range(1, w-1):
        for y in range(1, h-1):
            div[y, x] = -g(abs(gx[y, x])) * gx[y, x] - g(abs(gy[y, x])) * gy[y, x]
    ##print("VALEUR RETOUR" + str(div))
    return div

def anisotropic_diffusion(img, dt, num_iter):
    """Application de la diffusion anisotropique sur l'image."""
    u = img.astype(np.float32)
    print("IMAGE: " + str(u))
    ##print("TEST: " + str(u[1,0]))
    for i in range(num_iter):
        ##div = divergence(u)
        div = test(u)
        u = u + dt * div
    print("res: " + str(u))
    return np.clip(u, 0, 255).astype(np.uint8)


img = cv2.imread('mandrill-g0_5.png',cv2.IMREAD_GRAYSCALE)
##scale_percent = 110 # percent of original size
##width = int(img.shape[1] * scale_percent / 100)
##height = int(img.shape[0] * scale_percent / 100)
##dim = (width, height)
##  
### resize image
##resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
##
##

res = anisotropic_diffusion(img,0.1,90)
##res = anisotropic_diffusion(resized,0.1,18)
# concatenate image Horizontally
Hori = np.concatenate((img, res), axis=1)
cv2.imshow('Diffusion anisotropique', Hori)
##cv2.imshow('Diffusion anisotropique', res)