# Import image file, convert it to array and do some operations

import numpy as np
from PIL import Image

im = Image.open('image1.png')
print(im.size, im.mode)
pic = np.array(im)
print(pic.shape)
out1 = im.convert("L")
print(out1.size, out1.mode)
pic_grey_sclae = np.array(out1)
print(pic_grey_sclae.shape)
print(pic_grey_sclae)
im.show()
out1.show(title='GreyScale')

im = Image.open('image2.png')
print(im.size, im.mode)
im = im.resize((460,359))
pic = np.array(im)
print(pic.shape)
out2 = im.convert("L")
print(out2.size, out2.mode)
pic_grey_sclae = np.array(out2)
print(pic_grey_sclae.shape)
print(pic_grey_sclae)
im.show()
out2.show(title='GreyScale')

addition = np.add(out1,out2)
print(addition.shape)

final_image = Image.fromarray(addition, 'L')
final_image.show()