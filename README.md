# Faster RCNN
Faster R-CNN Resnet-101-FPN model was implemented with tensorflow 2.0.

# Requirements
- python 3.*
- tensorflow 2.* (tensorflow==2.0.0-alpha0)
- scikit-image
- cv2

# Training
## args
```
-b batch_size (default = 1)
```

```
-f flip_ratio (default = 0)
```

```
-l learning_rate (default = 1e-4)
```

## Command
``` python
python train_model.py -b batch_size -f flip_ratio -l learning_rate
```

# Acknowledgement
This work builds on many excellent works, which include:
- Heavily based on [tf-eager-fasterrcnn](https://github.com/Viredery/tf-eager-fasterrcnn)
- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
