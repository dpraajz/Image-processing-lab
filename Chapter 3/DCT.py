import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL.ImageFilter import BLUR, MedianFilter
from scipy.fftpack import dct


def dct2d(x,inverse=False):
    t = 2 if not inverse else 3
    temp = dct(x,type=t,norm='ortho').transpose()
    return dct(temp,type=t,norm='ortho').transpose()

im = Image.open('Lenna.png')
out = im.convert("L")

pic = np.array(out)
print(pic.shape)

new_pic = np.zeros((pic.shape))

for count,row in enumerate(pic):
    for num,each in enumerate(row):
        new_value = each*(-1)**(count+num)
        new_pic[count,num] = new_value

new_out = Image.fromarray(new_pic)
out.show()
cosine = dct2d(pic)
cosine_image = Image.fromarray(cosine)
cosine_image.show()
