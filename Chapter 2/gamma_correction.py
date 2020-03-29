import numpy as np
from PIL import Image

im = Image.open('washed_out_aerial_image.tif')
out = im.convert("L")
pic = np.array(out)
print(pic.shape)
# print(pic)

def gamma_correction(i):
    i_normalized = i/255
    i_corrected = i_normalized**(4)
    return i_corrected*255


new_pic = np.zeros((pic.shape))

for count,row in enumerate(pic):
    for num,each in enumerate(row):
        new_value = gamma_correction(each)
        new_pic[count,num] = new_value

final_image = Image.fromarray(new_pic)

print(new_pic[1])
out.show()
final_image.show()