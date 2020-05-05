import os
import numpy as np
import imageio

R_means = []
G_means = []
B_means = []

R_stds = []
G_stds = []
B_stds = []

filepath = r'./dataset/images'
pathDir = os.listdir(filepath)

for idx in range(len(pathDir)):
    print(idx, len(pathDir))

    filename = pathDir[idx]
    im = imageio.imread(os.path.join(filepath, filename)) / 255.0

    im_R = im[:,:,0]
    im_G = im[:,:,1]
    im_B = im[:,:,2]

    im_R_mean = np.mean(im_R)
    im_G_mean = np.mean(im_G)
    im_B_mean = np.mean(im_B)
    im_R_std = np.std(im_R)
    im_G_std = np.std(im_G)
    im_B_std = np.std(im_B)

    R_means.append(im_R_mean)
    G_means.append(im_G_mean)
    B_means.append(im_B_mean)
    R_stds.append(im_R_std)
    G_stds.append(im_G_std)
    B_stds.append(im_B_std)

a = [R_means,G_means,B_means]
b = [R_stds,G_stds,B_stds]

mean = [0,0,0]
std = [0,0,0]

mean[0] = np.mean(a[0])
mean[1] = np.mean(a[1])
mean[2] = np.mean(a[2])

std[0] = np.mean(b[0])
std[1] = np.mean(b[1])
std[2] = np.mean(b[2])

print('数据集的RGB平均值为\n[{},{},{}]'.format(mean[0],mean[1],mean[2]))
print('数据集的RGB方差为\n[{},{},{}]'.format(std[0],std[1],std[2]))
