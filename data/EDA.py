import glob
from PIL import Image
import os

path = r''
paths = glob.glob(path + '/**/*')
paths = [x for x in paths if os.path.isfile(x)]
print('loaded')
size = []
print('size\n', size)
ext = []
for i in paths:
    ext.append(os.path.splitext(i)[1])
    size.append(Image.open(i).size)

size = set(size)
ext = set(ext)
print('size/n', size)
print('ext/n', ext)