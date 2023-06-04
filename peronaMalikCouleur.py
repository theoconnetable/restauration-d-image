import numpy as np
import cv2

def g(x):
    """Tendency to decrease function on the interval [0, inf[."""
    return 1 / (1 + x**2)

def gradient_x(img):
    h, w, _ = img.shape
    gx = np.zeros((h, w, _))
    for y in range(h):
        for x in range(1, w-1):
            tab = gx[y,x]
            for c in range(_):
                tab[c] = (img[y, x+1, c] - img[y, x-1, c]) /2
            gx[x,y] = tab
    return gx

def gradient_y(img):
    h, w, _ = img.shape
    gy = np.zeros((h, w, _))
    for y in range(1, h-1):
        for x in range(w):
            tab = gy[y,x]
            for c in range(_):
                tab[c] = (img[y+1, x, c] - img[y-1, x, c]) / 2
            gy[y, x] = tab
    return gy

def divergence(img):
    h, w, _ = img.shape
    gx = gradient_x(img)
    gy = gradient_y(img)
    div = np.zeros((h, w, _))
    for y in range(h):
        for x in range(w):
            tab = div[y,x]
            for c in range(_):
                part1 = g(abs(gx[y, x, c])) * gx[y, x, c] - g(abs(gx[y, x, c])) * gx[y, x-1, c]
                part2 = g(abs(gx[y, x, c])) * gx[y, x, c] - g(abs(gx[y, x, c])) * gx[y-1, x, c]
                tab[c] = part1 + part2
            div[y, x] = tab
    return div

def anisotropic_diffusion(img, dt, num_iter):
    u = img.astype(np.float32)
    for i in range(num_iter):
        div = divergence(u)
        u = u + dt * div
    return np.clip(u, 0, 255).astype(np.uint8)

img = cv2.imread('mandrill-g0_5.png', cv2.IMREAD_COLOR)
res = anisotropic_diffusion(img, 0.25, 90)

scale_percent = 200
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

originImg = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
resImg = cv2.resize(res, dim, interpolation=cv2.INTER_AREA)

Hori = np.concatenate((originImg, resImg), axis=1)
cv2.imshow('Anisotropic Diffusion', Hori)
cv2.waitKey(0)
cv2.destroyAllWindows()
