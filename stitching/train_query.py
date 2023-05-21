#%%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
origin_path = "./ex.jpg"
ex_path = "./ex2.jpg"

#%%
img = cv.imread(origin_path)
ex_img = cv.imread(ex_path)

plt.imshow(img)
#%%
plt.imshow(ex_img)
#%%
# resize
width, height, _ = img.shape
ex_img = cv.resize(ex_img, (height, width))
plt.imshow(ex_img)

# %%
def SIFT(img):
    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc

# %%
kp1, desc1 = SIFT(img)
kp2, desc2 = SIFT(ex_img)

# %%
bf = cv.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

good_match = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_match.append([m])

#%%
img_pair = cv.drawMatchesKnn(img, kp1, ex_img, kp2, good_match[:10], None, flags=2)
plt.imshow(img_pair)

#%%
p1, p2 = matches[2]

print(f"p1: {p1.imgIdx, p1.queryIdx, p1.trainIdx}")
print(f"p2: {p2.imgIdx, p2.queryIdx, p2.trainIdx}")

# %%
