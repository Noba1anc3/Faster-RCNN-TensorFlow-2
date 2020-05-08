import os
import sys
import json
import getopt
import numpy as np
import tensorflow as tf

from tensorflow import keras

from detection.datasets import coco, data_generator
from detection.models.detectors import faster_rcnn

from pycocotool.cocoeval import COCOeval

# 1: all log info / 3: error log info only
# display warning and error log info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

assert tf.__version__.startswith('2.')

tf.random.set_seed(22)
np.random.seed(22)

# The mean and std values are decided by the pretrained models.
# When you are finetuning with some pretrained model,
# you need to follow the mean and std values used for pretraining.
img_mean = (123.675, 116.28, 103.53)
img_std = (58.395, 57.12, 57.375)

epochs = 100
batch_size = 2
flip_ratio = 0
learning_rate = 1e-4
checkpoint = 500
finetune = 1

opts, args = getopt.getopt(sys.argv[1:], "-b:-f:-l:-e:-c:-n:", )

for opt, arg in opts:
    if opt == '-b':
        batch_size = int(arg)
    elif opt == '-f':
        finetune = int(arg)
    elif opt == '-l':
        learning_rate = float(arg)
    elif opt == '-e':
        epochs = int(arg)
    elif opt == '-c':
        checkpoint = int(arg)
    elif opt == '-n':
        if int(arg) == 0:
            img_mean = (0., 0., 0.)
            img_std = (1., 1., 1.)
        elif int(arg) == 1:
            # Company Articles Dataset
            img_mean = (0.9684, 0.9683, 0.9683)
            img_std = (0.1502, 0.1505, 0.1505)

train_dataset = coco.CocoDataSet(dataset_dir='dataset', subset='train',
                                 flip_ratio=flip_ratio, pad_mode='fixed',
                                 mean=img_mean, std=img_std,
                                 scale=(800, 1216))
test_dataset = coco.CocoDataSet(dataset_dir='dataset', subset='val',
                                flip_ratio=flip_ratio, pad_mode='non-fixed',
                                mean=img_mean, std=img_std,
                                scale=(800, 1216))

train_generator = data_generator.DataGenerator(train_dataset)
train_tf_dataset = tf.data.Dataset.from_generator(
    train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.batch(batch_size).prefetch(100)#.shuffle(100)

num_classes = len(train_dataset.get_categories())
model = faster_rcnn.FasterRCNN(num_classes=num_classes)
# optimizer = keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)

if finetune:
    model.load_weights('model/faster_rcnn.h5')

for epoch in range(1, epochs, 1):

    for (batch, inputs) in enumerate(train_tf_dataset):
        print(batch)
        batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs

        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = \
                model((batch_imgs, batch_metas, batch_bboxes, batch_labels))

            loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if batch % 10 == 0 and not batch == 0:
            print('Epoch:', epoch, 'Batch:', batch, 'Loss:', loss_value.numpy(),
                  'RPN Class Loss:', rpn_class_loss.numpy(),
                  'RPN Bbox Loss:', rpn_bbox_loss.numpy(),
                  'RCNN Class Loss:', rcnn_class_loss.numpy(),
                  'RCNN Bbox Loss:', rcnn_bbox_loss.numpy())

        if batch % checkpoint == 0 and not batch == 0:
            model.save_weights('./model/epoch_' + str(epoch) + '_batch_' + str(batch) + '.h5')

            dataset_results = []
            imgIds = []

            for idx in range(len(test_dataset)):
                if idx % 10 == 9 or idx + 1 == len(test_dataset):
                    print(str(idx + 1) + ' / ' + str(len(test_dataset)))

                img, img_meta, _, _ = test_dataset[idx]

                proposals = model.simple_test_rpn(img, img_meta)
                res = model.simple_test_bboxes(img, img_meta, proposals)

                # visualize.display_instances(ori_img, res['rois'], res['class_ids'],
                #                             test_dataset.get_categories(), scores=res['scores'])

                image_id = test_dataset.img_ids[idx]
                imgIds.append(image_id)

                for pos in range(res['class_ids'].shape[0]):
                    results = dict()
                    results['score'] = float(res['scores'][pos])
                    results['category_id'] = test_dataset.label2cat[int(res['class_ids'][pos])]
                    y1, x1, y2, x2 = [float(num) for num in list(res['rois'][pos])]
                    results['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
                    results['image_id'] = image_id
                    dataset_results.append(results)

            if not dataset_results == []:
                with open('result/epoch_' + str(epoch) + '_batch_' + str(batch) + '.json', 'w') as f:
                    f.write(json.dumps(dataset_results))

                coco_dt = test_dataset.coco.loadRes(
                    'result/epoch_' + str(epoch) + '_batch_' + str(batch) + '.json')
                cocoEval = COCOeval(test_dataset.coco, coco_dt, 'bbox')
                cocoEval.params.imgIds = imgIds

                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
