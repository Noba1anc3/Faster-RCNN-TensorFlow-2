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
- [Project Design Documentation](https://github.com/Noba1anc3/Faster-RCNN-TensorFlow-2/wiki/Project-Design-Document)
- [Records on Debug & Addition of Original Project](https://github.com/Noba1anc3/Faster-RCNN-TensorFlow-2/wiki/%E5%AF%B9%E5%8E%9F%E6%9C%89%E9%A1%B9%E7%9B%AE%E7%9A%84%E8%B0%83%E8%AF%95%E7%BA%A0%E9%94%99%E5%92%8C%E6%94%B9%E9%80%A0%E6%B7%BB%E5%8A%A0%E8%AE%B0%E5%BD%95)
- [Visualized comparison between different normalization scheme](https://github.com/Noba1anc3/Faster-RCNN-TensorFlow-2/wiki/Comparison-between-different-normalization)
- [Detailed Training and Testing Logs of Training Without Normalization](https://github.com/Noba1anc3/Faster-RCNN-TensorFlow-2/wiki/Detailed-Training-and-Testing-Logs-of-Training-Without-Normalization)
- [Analysis on training result caused by different normalization scheme I](https://github.com/Noba1anc3/Faster-RCNN-TensorFlow-2/wiki/Analysis-on-training-result-caused-by-different-normalization-scheme-I)
- [Analysis on training result caused by different normalization scheme II ☆](https://github.com/Noba1anc3/Faster-RCNN-TensorFlow-2/wiki/%E2%98%86-Analysis-on-training-result-caused-by-different-normalization-scheme-II-%E2%98%86)
- [Analysis on training result caused by different anchor ratios](https://github.com/Noba1anc3/Faster-RCNN-TensorFlow-2/wiki/Analysis-on-training-result-caused-by-different-anchor-ratios)
- [Comparison with Detectron's Faster RCNN](https://github.com/Noba1anc3/Faster-RCNN-TensorFlow-2/wiki/Comparison-with-Detectron's-Faster-RCNN)

# Experiment
## Comparison between without normalization and ImageNet Normalization
![](http://m.qpic.cn/psc?/fef49446-40e0-48c4-adcc-654c5015022c/U9VSE8DftkGCrX.UXUSpmxIT4b**SQhrHn6NAn98RVNPQvml82nEWGkQemceMb78Y2pOnzhC.ocBsHnTfSQm0YjwcvdKn.Bc*g4RzGizWbc!/b&bo=TALgAUwC4AEDGTw!&rf=viewer_4)

## Comparison between Anchor Ratio of (0.5 1 2), (1 2 4) and (1 3 6)
![](https://camo.githubusercontent.com/dbb5fe86ec44bdd3e2bc5f3e1e25b489fee563eb/687474703a2f2f6d2e717069632e636e2f7073633f2f66656634393434362d343065302d343863342d616463632d3635346335303135303232632f393079664f2e38624f6164584545344d694873506e782e5367683761334e52634c34744e6b6c52554e6e566c7470794b654a5878733055796c785731767971713831786d4c7039666c4d42484b5575672e5a4f72677721212f6226626f3d54514c624155304332774544435377212672663d7669657765725f34)
![](https://camo.githubusercontent.com/507dabe208a16867b7b1f666f2f5edba93221e4f/687474703a2f2f6d2e717069632e636e2f7073633f2f66656634393434362d343065302d343863342d616463632d3635346335303135303232632f393079664f2e38624f6164584545344d694873506e393031344f6e386646495a746a6a4b6e545a466c7a496b61315068743242485065317743554b5330336937455a464d6e3476633852324e54385859636b4674326721212f6226626f3d50674c4e415434437a514544435377212672663d7669657765725f34)

## Comparison between TensorFlow2 FasterRCNN and PyTorch Detectron2 FasterRCNN
![](https://camo.githubusercontent.com/89c078237ffc4bc05f483fb24ff4aea27578ae75/687474703a2f2f6d2e717069632e636e2f7073633f2f66656634393434362d343065302d343863342d616463632d3635346335303135303232632f393079664f2e38624f6164584545344d694873506e77625341782a7830494c7a57574c4565453645557857534a45784b6c4e38687a6d645235793376346c4a697539623130592a4a68434e6c775a44514555784b717721212f6226626f3d4e674c4a4151414141414144423934212672663d7669657765725f34)

# Acknowledgement
This work builds on many excellent works, which include:
- Heavily based on [tf-eager-fasterrcnn](https://github.com/Viredery/tf-eager-fasterrcnn)
- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)


