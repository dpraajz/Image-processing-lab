import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL.ImageFilter import BLUR, MedianFilter


im = Image.open('rectangle.tif')
out = im.convert("L")

pic = np.array(out)
print(pic.shape)

new_pic = np.zeros((pic.shape))

for count,row in enumerate(pic):
    for num,each in enumerate(row):
        new_value = each*(-1)**(count+num)
        new_pic[count,num] = new_value

new_out = Image.fromarray(new_pic)

fourier = np.fft.fft2(out)
plt.figure(0)
plt.imshow(out)
plt.figure(1)
plt.imshow(abs(fourier))
plt.figure(2)
plt.imshow(abs(np.fft.fft2(new_out)))
plt.figure(3)
plt.imshow(1+np.log10(abs(fourier)))
plt.show()