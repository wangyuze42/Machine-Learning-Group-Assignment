#!/usr/bin/python3
# __*__ coding: utf-8 __*__
from scipy.ndimage import filters
from scipy.signal import convolve2d
from skimage import exposure
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False
def image2array(image):
    image = np.array(image)
    return image
def array2image(arrimg):
    image = Image.fromarray(arrimg.astype('uint8')).convert('RGB')
    return image


def image_mean_filter(pil_im):
    image_arr = image2array(pil_im)
    dst_arr = np.zeros_like(image_arr)
    mean_operator = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]) / 9
    if len(image_arr.shape) == 3:
        for i in range(3):
            dst_arr[:, :, i] = convolve2d(image_arr[:, :, i], mean_operator, mode="same")
    elif len(image_arr.shape) == 2:
        dst_arr = convolve2d(image_arr, mean_operator, mode="same")

    return array2image(dst_arr)

def image_medium_filter(pil_im, sigma=5):
    image_arr = image2array(pil_im)
    if len(image_arr.shape) == 3:
        for i in range(3):
           image_arr[:,:,i] = filters.median_filter(image_arr[:,:,i], sigma)
        return image_arr
    elif len(image_arr.shape) == 2:
        image_arr = filters.median_filter(image_arr, sigma)
        return array2image(image_arr)

def image_laplace_filter(pil_im,sigma):
    image_arr = np.array(pil_im)
    dst_arr = np.zeros(image_arr.shape, dtype=np.uint8)
    filter_arr = dst_arr
    laplace_operator = np.array([[0, -1, 0],
                                [-1, 4, -1],
                                [0, -1, 0]])
    if len(image_arr.shape) == 3:
        for i in range(3):
            dst_arr[:,:,i] = convolve2d(image_arr[:,:,i], laplace_operator, mode="same")
            filter_arr[:,:,i] = filters.gaussian_filter(dst_arr[:,:,i], sigma)
    elif len(image_arr.shape) == 2:
        dst_arr = convolve2d(image_arr, laplace_operator, mode="same")
        filter_arr = filters.gaussian_filter(dst_arr, sigma)
    dst_arr = image_arr + filter_arr
    dst_arr = dst_arr / 255.0
    # 饱和处理
    mask_1 = dst_arr  < 0
    mask_2 = dst_arr  > 1
    dst_arr = dst_arr * (1-mask_1)
    dst_arr = dst_arr * (1-mask_2) + mask_2
    return array2image(dst_arr*255)

if __name__ == '__main__':
    image = Image.open('../input/group_project/train_big/Achillea maritima - Google Search/005.jpg').convert('L')
    image1 = image_mean_filter(image)
    image2 = image_medium_filter(image)
    image3 = image_laplace_filter(image,sigma=5)
    plt.figure()
    plt.subplot(221)
    plt.imshow(image1)
    plt.subplot(222)
    plt.imshow(image2)
    plt.subplot(223)
    plt.imshow(image3)
    plt.show()
