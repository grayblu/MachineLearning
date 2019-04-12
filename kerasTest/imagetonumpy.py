from PIL import Image
import numpy as np

im = Image.open('test.png')

im2arr = np.array(im)