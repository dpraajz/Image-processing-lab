import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from PIL.ImageOps import equalize


def contraharmonic_mean_filter_2d(img, filter,Q):
    img = img
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
            prod = np.multiply(img_portion, filter)

            res_1 = np.sum(prod**(Q+1))
            res_2 = np.sum(prod**(Q))
            res = res_1/res_2
            final_matrix[i+1][j+1] = res


    result_matrix = final_matrix[m-1:-(m-1),n-1:-(n-1)] 
    # result_matrix = result_matrix * (255/result_matrix.max())
    return result_matrix


def pepper_noise(image,prob):
    '''
    Add pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    # thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            else:
                output[i][j] = image[i][j]
    return output

def salt_noise(image,prob):
    '''
    Add salt noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

im = Image.open('ckt-board-orig.tif')
out = im.convert("L")
img = np.array(im)
histogram_original = out.histogram()
out.show()

salt_noisy = salt_noise(img,0.01) 
salt_noisy_im = Image.fromarray(salt_noisy)
salt_noisy_im.show()

pepper_noisy = pepper_noise(img,0.01)
pepper_noisy_im = Image.fromarray(pepper_noisy)
pepper_noisy_im.show()

contraharmonic_mean_filter = np.ones((3,3)) 

contraharmonic_mean_filter_image_for_pepper_array = contraharmonic_mean_filter_2d(pepper_noisy_im, contraharmonic_mean_filter, 1.5)
contraharmonic_filtered_image_pepper = Image.fromarray(contraharmonic_mean_filter_image_for_pepper_array)
contraharmonic_filtered_image_pepper.show()


contraharmonic_mean_filter_image_for_salt_array = contraharmonic_mean_filter_2d(salt_noisy_im, contraharmonic_mean_filter, -1.5)
contraharmonic_filtered_image_salt = Image.fromarray(contraharmonic_mean_filter_image_for_salt_array)
contraharmonic_filtered_image_salt.show()

