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



def median_filter_2d(img, filter):
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

            res = np.median(np.multiply(img_portion, filter))
            final_matrix[i+1][j+1] = res


    result_matrix = final_matrix[m-1:-(m-1),n-1:-(n-1)]
    return result_matrix




im = Image.open('ckt-board-orig.tif')
out = im.convert("L")
img = np.array(im)
out.show()


salt_and_pepper_noisy = sp_noise(img,0.1)
salt_and_pepper_noisy_im = Image.fromarray(salt_and_pepper_noisy)
salt_and_pepper_noisy_im.show()
salt_and_pepper_noisy_histogram = salt_and_pepper_noisy_im.histogram()

arithematic_median_filter = np.ones((3,3))
median_filtered_image_array = median_filter_2d(salt_and_pepper_noisy,arithematic_median_filter)
median_filtered_image = Image.fromarray(median_filtered_image_array)
median_filtered_image.show()

# after two more passes

for i in range(0,2):
    median_filtered_image_array = median_filter_2d(median_filtered_image_array,arithematic_median_filter)

median_filtered_image = Image.fromarray(median_filtered_image_array)
median_filtered_image.show()
