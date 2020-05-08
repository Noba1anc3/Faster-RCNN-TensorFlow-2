import visualize
import numpy as np

from matplotlib import pyplot as plt

from detection.datasets import coco

img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)

batch_size = 1
flip_ratio = 0

train_dataset = coco.CocoDataSet(dataset_dir='dataset', subset='train',
                                 flip_ratio=flip_ratio, pad_mode='non-fixed',
                                 mean=img_mean, std=img_std,
                                 scale=(800, 1216))

img, img_meta, bboxes, labels = train_dataset[0]
rgb_img = np.round(img*img_std + img_mean)
visualize.display_instances(rgb_img, bboxes, labels, train_dataset.get_categories())

# plt.savefig('img_demo.png')
