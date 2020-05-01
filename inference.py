import os
import json
import numpy as np
import tensorflow as tf

from pycocotools.cocoeval import COCOeval
from detection.datasets import coco, data_generator
from detection.models.detectors import faster_rcnn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

assert tf.__version__.startswith('2.')

tf.random.set_seed(22)
np.random.seed(22)

img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)

batch_size = 1
flip_ratio = 0

train_dataset = coco.CocoDataSet(dataset_dir='dataset', subset='train',
                                 flip_ratio=flip_ratio, pad_mode='fixed',
                                 mean=img_mean, std=img_std,
                                 scale=(800, 1216))

train_generator = data_generator.DataGenerator(train_dataset)
train_tf_dataset = tf.data.Dataset.from_generator(
    train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.batch(batch_size).prefetch(100).shuffle(100)

num_classes = len(train_dataset.get_categories())
model = faster_rcnn.FasterRCNN(num_classes=num_classes)

img, img_meta, _, _ = train_dataset[0]
batch_imgs = tf.convert_to_tensor(np.expand_dims(img, 0))  # [1, 1216, 1216, 3]
batch_metas = tf.convert_to_tensor(np.expand_dims(img_meta, 0))  # [1, 11]

_ = model((batch_imgs, batch_metas), training=False)
model.load_weights('model/epoch_10.h5', by_name=True)

dataset_results = []
imgIds = []

for idx in range(len(train_dataset)):
    if idx % 10 == 9 or idx + 1 == len(train_dataset):
        print(str(idx+1) + ' / ' + str(len(train_dataset)))

    img, img_meta, _, _ = train_dataset[idx]

    proposals = model.simple_test_rpn(img, img_meta)
    res = model.simple_test_bboxes(img, img_meta, proposals)
    # visualize.display_instances(ori_img, res['rois'], res['class_ids'],
    #                             train_dataset.get_categories(), scores=res['scores'])

    image_id = train_dataset.img_ids[idx]
    imgIds.append(image_id)
    print(res)
    for pos in range(res['class_ids'].shape[0]):
        results = dict()
        results['score'] = float(res['scores'][pos])
        results['category_id'] = train_dataset.label2cat[int(res['class_ids'][pos])]
        y1, x1, y2, x2 = [float(num) for num in list(res['rois'][pos])]
        results['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
        results['image_id'] = image_id
        dataset_results.append(results)


with open('detection_result.json', 'w') as f:
    f.write(json.dumps(dataset_results))

coco_dt = train_dataset.coco.loadRes('detection_result.json')
cocoEval = COCOeval(train_dataset.coco, coco_dt, 'bbox')
cocoEval.params.imgIds = imgIds

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
