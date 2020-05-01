import visualize
import numpy as np

from matplotlib import pyplot as plt

from detection.datasets import coco
from detection.datasets.utils import get_original_image

img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)

batch_size = 1
flip_ratio = 0

train_dataset = coco.CocoDataSet(dataset_dir='dataset', subset='train',
                                 flip_ratio=flip_ratio, pad_mode='fixed',
                                 mean=img_mean, std=img_std,
                                 scale=(800, 1216))

img, img_meta, bboxes, labels = train_dataset[0]
rgb_img = np.round(img + img_mean)
visualize.display_instances(rgb_img, bboxes, labels, train_dataset.get_categories())

plt.savefig('img_demo.png')


array1 = []
for i in range(100):
    array1.append(i)
array1.insert(0, 797)
array1.remove(99)

array2 = [399,  599,  798,  997, 1715, 1716, 1717, 1718, 1719, 1720, 1723,
       1724, 1725, 1726, 1727, 1728, 1731, 1732, 1733, 1734, 1735, 1736,
       1739, 1740, 1741, 1742, 1743, 1744, 1747, 1748, 1749, 1750, 1751,
       1752, 1753, 1769, 1770, 1869, 1870, 1871, 1872, 1873, 1874, 1875,
       1892, 1893, 1894, 1895, 1896, 1898, 1899, 1900, 1901, 1902, 1903,
       1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914,
       1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925,
       1927, 1929, 1930]

import tensorflow as tf
array1 = tf.convert_to_tensor(array1,dtype=tf.int64)
array2 = tf.convert_to_tensor(array2,dtype=tf.int64)
print([array1,array2])
array = tf.concat([array1, array2], axis=0)
print(array)