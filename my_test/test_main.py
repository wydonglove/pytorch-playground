import os
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as pyplot
import cv2
import math

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

path = "cifar-10-batches-py/test_batch"

dict = unpickle(path)


# for key,item in dict.items():
#     print(key)
#     print(item)

data = dict[b'data']
print(data.shape)

data0 = data[2]
print(data0.shape)
#print(data0)

r = []
g = []
b = []
for x in range(1024):
    r.append(data0[x])
    g.append(data0[1024+x])
    b.append(data0[2048 + x])

n_data = []
for index in range(1024):
    n_data.append(r[index])
    n_data.append(g[index])
    n_data.append(b[index])

data0_to34 =  np.array(n_data).reshape(3,1024)
#n_datax =  np.array(data0_to34).reshape()
data0_to3 = np.array(n_data).reshape(32,32,3)
print(data0_to3)
data0_to3 = data0_to3[...,::-1]
cv2.imwrite("test_0.png",data0_to3)

print("-----------")
data0_to3 = np.array(data0).reshape(3,32,32)
r = Image.fromarray(data0_to3[0]).convert('L')
g = Image.fromarray(data0_to3[1]).convert('L')
b = Image.fromarray(data0_to3[2]).convert('L')
image = Image.merge("RGB", (r, g, b))

# 显示图片
# pyplot.imshow(image)
# pyplot.show()

#image.save("test_1.png", 'png')

print(data0_to3)
print("+++++")
print(np.array(data0_to3).transpose(1, 2, 0))

# xx = -1/3 * math.log2(1/3) - 2/3 * math.log2(2/3)
# xx1 = -1/5 * math.log2(1/5) - 4/5 * math.log2(4/5)
# print(xx,xx1)




from utee import misc
from collections import Counter

pkl_path = "../tmp/public_dataset/pytorch/imagenet-data/"
d = misc.load_pickle(pkl_path+'val224.pkl')
data = d['data']
target = d['target']
print(len(data),len(target))
print(data[1].shape)
print(target)
result = Counter(target)
print(result)

for index,item in enumerate(data):
    if target[index] == 823:
        print("---")
        dt = np.array(item).transpose(1, 2, 0)
        cv2.imwrite("imagenet_"+str(index)+".png", dt)

# data2 = np.array(data[5]).transpose(1, 2, 0)
# print(data2.shape)
# cv2.imwrite("imagenet_5.png",data2)
# target = d['target']
# result = Counter(target)
# print(result)