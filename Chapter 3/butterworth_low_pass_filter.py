import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL.ImageFilter import BLUR, MedianFilter
from scipy.ndimage import gaussian_filter, fourier_gaussian
from math import pow, exp

im = Image.open('characters_test_pattern.tif')
out = im.convert("L")
# out.show()
f = np.array(out)
# 1. Given an input image f (x, y) of size M * N, obtain the padding parameters P and Q . 
# Typically, we select P = 2M and Q = 2N.

M = f.shape[0]
N = f.shape[1]
P = 2*M
Q = 2*N
print(M,N,P,Q)

# 2. Form a padded image, fp(x, y),  of size P * Q by appending the 
# necessary number of zeros to f (x, y).
fp = np.zeros((P,Q))
fp[:f.shape[0],:f.shape[1]] = f
fp_pic = Image.fromarray(fp)
# fp_pic.show()
print(fp.shape)

# 3. Multiply fp(x, y) by ( -1)^ {x + y} to center its transform.
fp_centered = np.zeros((P,Q))
for count,row in enumerate(fp):
    for num,each in enumerate(row):
        new_value = each*(-1)**(count+num)
        fp_centered[count,num] = new_value
fp_centered_pic = Image.fromarray(fp_centered)
# fp_centered_pic.show()

# 4. Compute the DFT, F (u, v), of the image from step 3.

F = np.fft.fft2(fp_centered)
# plt.figure(0)
# plt.imshow(1+np.log10(abs(F)),cmap='Greys',interpolation='nearest')
# plt.show()

# 5 .Generate a real, symmetric filter function, H(u, v), of size P *Q with center at 
# coordinates (P/2, Q/2).  Form the product G(u, v) =H(u, v)F(u, v) using array multiplication; 
# that is, G(i, k) = H(i, k)F(i, k). then obtain the processed image

R = 460
n = 2 # order
# H = np.ones((P,Q))
H = np.zeros((P,Q))
for i in range (0,P):
    for j in range (0,Q):
        temp = (float(float(i) - float(P/2))**2 + float(float(j) - float(Q/2))**2)**(1/2)
        temp_1 = float((temp/R)**(2*n))
        temp_2 = 1 + temp_1
        H[i][j] = float (1/ temp_2)
 
        
g = np.multiply(F,H)

# plt.figure(0)
# plt.imshow((abs(H)),cmap='Greys',interpolation='nearest')
# plt.show()

# 6 Obtain the processed image using IDFT and recentering it

g_idft = np.fft.ifft2(g)
gp = np.zeros((P,Q))
for count,row in enumerate(g_idft):
    for num,each in enumerate(row):
        new_value = (each.real)*(-1)**(count+num)
        gp[count,num] = new_value

gp_image = Image.fromarray(abs(gp))
# gp_image.show() 

# 7 Obtain the final processed result, g (x, y), by extracting the M * N region from 
# the top, left quadrant of g p (x, y).
g = np.zeros((M,N))

for i in range(0,M):
    for j in range(0,N):
        g[i][j] = gp[i][j]

final_image = Image.fromarray(g)
final_image.show()