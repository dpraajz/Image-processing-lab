import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from PIL.ImageOps import equalize


def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

im = Image.open('original_pattern.tif')
out = im.convert("L")
img = np.array(im)
histogram_original = out.histogram()
out.show()

mean = 0
var = 10
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (256, 256)) 
noisy_image = np.zeros(img.shape, np.float32)

if len(img.shape) == 2:
    noisy_image = img + gaussian
else:
    noisy_image[:, :, 0] = img[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img[:, :, 2] + gaussian


gausian_noisy_im = Image.fromarray(noisy_image)
gausian_noisy_im.show()
gausian_noisy_histogram = gausian_noisy_im.histogram()

salt_and_pepper_noisy = sp_noise(img,0.02)
salt_and_pepper_noisy_im = Image.fromarray(salt_and_pepper_noisy)
salt_and_pepper_noisy_im.show()
salt_and_pepper_noisy_histogram = salt_and_pepper_noisy_im.histogram()

# Original histogram
plt.figure(0)
for i in range(0, 256):

    plt.bar(i, histogram_original[i], alpha=0.3)

# gausian noisy histogram
plt.figure(1)
for i in range(0, 256):

    plt.bar(i, gausian_noisy_histogram[i], alpha=0.3)

# gausian noisy histogram
plt.figure(2)
for i in range(0, 256):

    plt.bar(i, salt_and_pepper_noisy_histogram[i], alpha=0.3)

plt.show()
