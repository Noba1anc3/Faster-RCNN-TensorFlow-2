import os
import sys
import getopt
import numpy as np
import tensorflow as tf
from tensorflow import keras

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

epochs = 100
batch_size = 1
flip_ratio = 0
learning_rate = 1e-4

opts, args = getopt.getopt(sys.argv[1:], "-b:-f:-l:", )

for opt, arg in opts:
    if opt == '-b':
        batch_size = int(arg)
    elif opt == '-f':
        flip_ratio = float(arg)
    elif opt == '-l':
        learning_rate = float(arg)
    elif opt == '-e':
        epochs = int(arg)

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
optimizer = keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)

for epoch in range(epochs):
    loss_history = []

    for (batch, inputs) in enumerate(train_tf_dataset):
        batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs

        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = \
                model((batch_imgs, batch_metas, batch_bboxes, batch_labels))

            loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_history.append(loss_value.numpy())

        if batch % 10 == 0:
            print('Epoch:', epoch + 1, 'Batch:', batch, 'Loss:', np.mean(loss_history))

    model.save_weights('./model/epoch_' + str(epoch+1) + '.h5')
