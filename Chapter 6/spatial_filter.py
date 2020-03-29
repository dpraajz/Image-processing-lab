import numpy as np


def correlation_2d(img, filter):
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


# a = np.zeros((5,5))

# a[2,2] = 1


# filter = np.ones((3,3)) 
# num = 1
# for row_count,row in enumerate(filter):
#     for col_cout, val in enumerate(row):
#         filter[row_count,col_cout] = num
#         num += 1


        
# print(correlation_2d(a, filter))