from PIL import Image
import numpy as np

def image2array(image):
    image = np.array(image)
    return image

def array2image(arrimg):
    image = Image.fromarray(arrimg.astype('uint8')).convert('RGB')
    return image


def convert2gray(img):
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = r  # 可以是r,也可以是g，b
        return gray
    else:
        return img


if __name__ == '__main__':
    image = Image.open("../input/group_project/train_big/Achillea maritima - Google Search/005.jpg")
    image = image2array(image)
    image = convert2gray(image)
    image = array2image(image)
    image.show()
