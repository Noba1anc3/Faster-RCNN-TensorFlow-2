# Faster RCNN
Faster R-CNN Resnet-101-FPN model was implemented with tensorflow 2.0.

# Requirements
- python 3.*
- tensorflow>=2.2.0rc3
- scikit-image
- Keras==2.3.1
- cv2

# Training
## args
```
-b batch_size (default = 2)
```

```
-f flip_ratio (default = 0)
```

```
-l learning_rate (default = 1e-4)
```

```
-e epochs (default = 100)
```

```
-c checkpoint (default = 1 in train_epoch, default = 500 in train_batch)
```

```
-n normalization (default = ImageNet's mean and std.)
   n = 0: no normalization;
   n = 1: Company Articles Dataset's mean and std.
```

## Command
``` python
python train_batch.py [commands]
```

# Acknowledgement
This work builds on many excellent works, which include:
- Heavily based on [tf-eager-fasterrcnn](https://github.com/Viredery/tf-eager-fasterrcnn)
- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
