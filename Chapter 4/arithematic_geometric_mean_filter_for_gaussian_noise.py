import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from PIL.ImageOps import equalize


def mean_filter_2d(img, filter):
    m = np.shape(filter)[0]
    n = np.shape(filter)[1]
    padded_image =  np.pad(img,(m-1,n-1),'constant', constant_values=(0))

    # print(padded_image)

    m_padded_image = np.shape(padded_image)[0]
    n_padded_image = np.shape(padded_image)[1]

    row_length_loop = m_padded_image-(m-1)
    col_length_loop = n_padded_image-(n-1)

    final_matrix = np.zeros((m_padded_image,n_padded_image))
    for i in range(row_length_loop):
        for j in range(col_length_loop):
            img_portion = padded_image[i:i+m, j:j+n]

            res = np.sum(np.multiply(img_portion, filter))
            final_matrix[i+1][j+1] = res


    result_matrix = final_matrix[m-1:-(m-1),n-1:-(n-1)]
    return result_matrix


def geometric_mean_filter_2d(img, filter):
    m = np.shape(filter)[0]
    n = np.shape(filter)[1]
    padded_image =  np.pad(img,(m-1,n-1),'constant', constant_values=(0))

    # print(padded_image)

    m_padded_image = np.shape(padded_image)[0]
    n_padded_image = np.shape(padded_image)[1]

    row_length_loop = m_padded_image-(m-1)
    col_length_loop = n_padded_image-(n-1)

    final_matrix = np.zeros((m_padded_image,n_padded_image))
    for i in range(row_length_loop):
        for j in range(col_length_loop):
            img_portion = padded_image[i:i+m, j:j+n]

            res = (np.prod(np.multiply(img_portion, filter)))**(1/(m*n))
            final_matrix[i+1][j+1] = res


    result_matrix = final_matrix[m-1:-(m-1),n-1:-(n-1)]
    return result_matrix


im = Image.open('ckt-board-orig.tif')
out = im.convert("L")
img = np.array(im)
histogram_original = out.histogram()
# out.show()

mean = 10
var = 100
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (448, 464)) 
noisy_image = np.zeros(img.shape, np.float32)

if len(img.shape) == 2:
    noisy_image = img + gaussian
else:
    noisy_image[:, :, 0] = img[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img[:, :, 2] + gaussian


gausian_noisy_im = Image.fromarray(noisy_image)
gausian_noisy_im.show()

arithematic_mean_filter = np.ones((3,3))*(1/9)
mean_filtered_image_array = mean_filter_2d(img,arithematic_mean_filter)
mean_filtered_image = Image.fromarray(mean_filtered_image_array)
mean_filtered_image.show()


geometric_mean_filter = np.ones((3,3))
geometric_filtered_image_array = geometric_mean_filter_2d(img,geometric_mean_filter)
geometric_filtered_image = Image.fromarray(geometric_filtered_image_array)
geometric_filtered_image.show()