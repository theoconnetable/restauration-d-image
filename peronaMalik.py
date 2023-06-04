import numpy as np
import cv2

def g(x):
    """Fonction de tendance à diminuer sur l'intervalle [0, inf[."""
    return 1 / (1 + x**2)

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

def divergence(img):
    h, w = img.shape[:2]
    gx = gradient_x(img)
    gy = gradient_y(img)
    div = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            part1 = g(abs(gx[y,x])) * gx[y,x] - g(abs(gx[y,x])) * gx[y,x-1]
            part2 = g(abs(gx[y,x])) * gx[y,x] - g(abs(gx[y,x])) * gx[y-1,x]
            div[y,x] = part1 + part2
    return div ## +1 pas+1/2 ni -1
##Pas à rajouter les 1/2 car val dejà dans tab g(x,y) et g(x-1,y)



def anisotropic_diffusion(img, dt, num_iter):
    """Application de la diffusion anisotropique sur l'image."""
    u = img.astype(np.float32)
    print("IMAGE: " + str(u))
    for i in range(num_iter):
        div = divergence(u)
        u = u + dt * div
    return np.clip(u, 0, 255).astype(np.uint8)


img = cv2.imread('mandrill-g0_5.png',cv2.IMREAD_GRAYSCALE)
res = anisotropic_diffusion(img,0.25,180)

scale_percent = 200 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
originImg = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
resImg = cv2.resize(res, dim, interpolation = cv2.INTER_AREA)

# concatenate image Horizontally
Hori = np.concatenate((originImg, resImg), axis=1)
cv2.imshow('Diffusion anisotropique', Hori)
##cv2.imshow('Diffusion anisotropique', res)
