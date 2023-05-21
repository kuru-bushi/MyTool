# conda info name = stitching
# 参考
    - https://pystyle.info/opencv-feature-matching/

# trainIdx, queryIdx について
- distance: 特徴量記述子の距離
- queryIdx: クエリ記述子 (match(desc1, desc2) と渡した場合、desc1 のインデックス)
- trainIdx: 学習記述子 (match(desc1, desc2) と渡した場合、desc2 のインデックス)

# キーワード
- homography scratch optimize caribration

# 参考
- calibration
    - https://towardsdatascience.com/camera-calibration-with-example-in-python-5147e945cdeb
    - https://towardsdatascience.com/understanding-homography-a-k-a-perspective-transformation-cacaed5ca17
- 