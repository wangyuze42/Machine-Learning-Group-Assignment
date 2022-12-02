#!/usr/bin/python3
# __*__ coding: utf-8 __*__
import numpy as np
import cv2 as cv
import math
import os


# filePath = '../input/group_project/train_big/Achillea maritima - Google Search/'
# os.listdir(filePath)
def getfilename(filename):
    for root, dirs, files in os.walk(filename):
        array = dirs
        if array:
            return array


def bi_linear(src, dst, target_size):
    pic = cv.imread(src)
    th, tw = target_size[0], target_size[1]
    emptyImage = np.zeros(target_size, np.uint8)
    for k in range(3):
        for i in range(th):
            for j in range(tw):
                # 首先找到在原图中对应的点的(X, Y)坐标
                corr_x = (i + 0.5) / th * pic.shape[0] - 0.5
                corr_y = (j + 0.5) / tw * pic.shape[1] - 0.5
                point1 = (math.floor(corr_x), math.floor(corr_y))  # 左上角的点
                point2 = (point1[0], point1[1] + 1)
                point3 = (point1[0] + 1, point1[1])
                point4 = (point1[0] + 1, point1[1] + 1)

                fr1 = (point2[1] - corr_y) * pic[point1[0], point1[1], k] + (corr_y - point1[1]) * pic[
                    point2[0], point2[1], k]
                fr2 = (point2[1] - corr_y) * pic[point3[0], point3[1], k] + (corr_y - point1[1]) * pic[
                    point4[0], point4[1], k]
                emptyImage[i, j, k] = (point3[0] - corr_x) * fr1 + (corr_x - point1[0]) * fr2

    cv.imwrite(dst, emptyImage)


def main():
    a = getfilename('E:/semester1/scalable computing/lab/CS7NS1_Assignment_1/input/group_project/train_big/')
    # print(a)
    fname = []
    pictures = []
    for name in a:
        fname.append('E:/semester1/scalable computing/lab/CS7NS1_Assignment_1/input/group_project/train_big/' + name)

    for file in fname:
        for filename in os.listdir(file):
            # dir = file.replace('E:/semester1/scalable computing/lab/CS7NS1_Assignment_1/input/group_project','')
            # print(dir)
            a = file.replace(' - Google Search', '')
            a = a.replace('train_big', 'process_rescale')
            # print(dir)
            if not os.path.isdir(a):
                os.mkdir(a)
                print(a)
            pictures.append(file + '/' + filename)

    for picture in pictures:
        # src = '../input/group_project/train_big/Achillea maritima - Google Search/005.jpg'
        src = picture
        if src.endswith('.jpg'):
            trans = src.replace('train_big', 'process_rescale')
            trans = trans.replace(' - Google Search', '')
            # dst = '../input/group_project/process_rescale/Achillea maritima/005.jpg'
            dst = trans
            print(dst)
            target_size = (300, 300, 3)  # 变换后的图像大小
            try:
                bi_linear(src, dst, target_size)
            except:
                cv.imwrite(dst, cv.imread(src))


if __name__ == '__main__':
    main()
