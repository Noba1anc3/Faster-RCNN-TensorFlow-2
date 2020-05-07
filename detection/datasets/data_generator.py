import numpy as np


class DataGenerator(object):
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle

    def __call__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            pass
            # np.random.shuffle(indices)

        # bbox和label进行零填充，使得整个batch中的维度相同
        # 首先计算整个batch中每张图片最多拥有多少个bbox

        # num_bbox = []
        # for img_idx in indices:
        #     _, _, bbox, _ = self.dataset[img_idx]
        #     num_bbox.append(bbox.shape[0])
        # max_num_bbox = max(num_bbox)

        max_num_bbox = 45

        for img_idx in indices:
            img, img_meta, bbox, label = self.dataset[img_idx]

            # 填充bbox和label
            num_to_fill = max_num_bbox - bbox.shape[0]  # 需要填充多少个bbox和label
            bbox = np.concatenate([bbox, np.zeros((num_to_fill, 4), dtype=int)], axis=0)
            label = np.concatenate([label, np.zeros((num_to_fill,), dtype=int)], axis=0)

            yield img, img_meta, bbox, label
