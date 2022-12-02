import cv2
import matplotlib.pyplot as plt
import numpy
import skimage

plt.rc("font", family='Microsoft YaHei')
img = plt.imread('../input/group_project/final_aug/Anacamptis morio/006.jpg')

plt.subplot(221), plt.title('original')
plt.imshow(img, 'gray')
noise_img = skimage.util.random_noise(img, mode='salt') * 255
plt.subplot(222), plt.title('noise inside')
plt.imshow(noise_img, 'gray')

mean_img = noise_img

for i in range(1, noise_img.shape[0] - 1):
    for j in range(1, noise_img.shape[1] - 1):
        tmp = 0
        for k in range(-1, 2):
            for l in range(-1, 2):
                tmp += noise_img[i + k][j + l]
        mean_img[i][j] = tmp / 9
plt.subplot(223), plt.title('after')
plt.imshow(mean_img, 'gray')

median_img = noise_img

plt.show()
