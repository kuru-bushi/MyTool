#%%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
origin_path = "./ro1.jpg"
ex_path = "./ro2.jpg"

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
# get key point, descriptors
kp1, desc1 = SIFT(img)
kp2, desc2 = SIFT(ex_img)

# %%
bf = cv.BFMatcher()
select_match_type = "match" # knn, match
select_filter = True

def get_good_match(matches, filter=True, thredhold=0.7):
    if filter:
        good_match = []

        for m, n in matches:
            if m.distance < thredhold * n.distance:
                good_match.append([m])
        return good_match
    else:
        return matches

#%%
if select_match_type == "match":
    matches = bf.match(desc1, desc2)
    # 
    good_match = matches

    good_match = sorted(good_match, key = lambda x: x.distance)

    img_pair = cv.drawMatches(img, kp1, ex_img, kp2, good_match[:10], None, flags=2)
    print(len(good_match))

elif select_match_type == "knn":
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_match = get_good_match(matches, filter=select_filter)

    good_match = sorted(good_match, key = lambda x: x[0].distance)
    img_pair = cv.drawMatchesKnn(img, kp1, ex_img, kp2, good_match[:10], None, flags=2)
    print(len(good_match))

plt.imshow(img_pair)
#%%
# test
# mat0, mat1 = matches[2]
# print(mat0.imgIdx, mat0.trainIdx, mat0.queryIdx)
# print(mat1.imgIdx, mat1.trainIdx, mat1.queryIdx)

# %%
print(kp1[0].pt)

# %%
