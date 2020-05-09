# Faster RCNN
Faster R-CNN Resnet-101-FPN implementation based on TensorFlow 2.0.

# Requirements
- python 3.*
- tensorflow>=2.2.0rc3
- scikit-image
- cv2

# Training
## args
```
-b batch_size (default = 2)
```

```
-f finetune (default = 0)
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

# Wiki
- [Visualized comparison between different normalization scheme](https://github.com/Noba1anc3/Faster-RCNN-TensorFlow-2/wiki/Comparison-between-different-normalization)
- [Detailed Training and Testing Logs of Training Without Normalization](https://github.com/Noba1anc3/Faster-RCNN-TensorFlow-2/wiki/Detailed-Training-and-Testing-Logs-of-Training-Without-Normalization)
- [Analysis on training result caused by different normalization scheme I](https://github.com/Noba1anc3/Faster-RCNN-TensorFlow-2/wiki/Analysis-on-training-result-caused-by-different-normalization-scheme-I)
- [Analysis on training result caused by different normalization scheme II ☆](https://github.com/Noba1anc3/Faster-RCNN-TensorFlow-2/wiki/Analysis-on-training-result-caused-by-different-normalization-scheme-II)

# Experiment
## Comparison between without normalization and ImageNet Normalization
![](http://m.qpic.cn/psc?/fef49446-40e0-48c4-adcc-654c5015022c/U9VSE8DftkGCrX.UXUSpmxIT4b**SQhrHn6NAn98RVNPQvml82nEWGkQemceMb78Y2pOnzhC.ocBsHnTfSQm0YjwcvdKn.Bc*g4RzGizWbc!/b&bo=TALgAUwC4AEDGTw!&rf=viewer_4)

# Acknowledgement
This work builds on many excellent works, which include:
- Heavily based on [tf-eager-fasterrcnn](https://github.com/Viredery/tf-eager-fasterrcnn)
- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)


