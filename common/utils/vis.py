import cv2
import numpy as np
import matplotlib.pyplot as plt

def vis_keypoints(img, kps, skeleton, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(skeleton)):
        i1 = skeleton[l][0]
        i2 = skeleton[l][1]
        p1 = kps[i1,0].astype(np.int32), kps[i1,1].astype(np.int32)
        p2 = kps[i2,0].astype(np.int32), kps[i2,1].astype(np.int32)
        cv2.line(
            kp_mask, p1, p2,
            color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(
            kp_mask, p1,
            radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(
            kp_mask, p2,
            radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

