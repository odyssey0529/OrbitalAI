# OrbitalAI_TelePIX+KIOST

# BiseNetv2-Tensorflow
Use tensorflow to implement a real-time scene image segmentation
model based on paper "BiSeNet V2: Bilateral Network with Guided 
Aggregation for Real-time Semantic Segmentation". You may refer
https://arxiv.org/abs/2004.02147 for details.

The main network architecture is as follows:

`Network Architecture`
![NetWork_Architecture](./data/source_image/network_architecture.png)

## Installation
This software has only been tested on ubuntu 16.04(x64), python3.5, 
cuda-9.0, cudnn-7.0 with a GTX-1070 GPU. To use this repo you 
need to install tensorflow-gpu 1.15.0 and other version of 
tensorflow has not been tested but I think it will be able to 
work properly if new version was installed in your local machine. Other required 
package can be installed by

```
pip3 install -r requirements.txt
```

## CityScapes Dataset preparation
Once you have prepared the dataset's image well you may generate
the training image index file by
```
python ./data/example_dataset/cityscapes/image_file_index/make_image_file_index.py
```


## Test model
You can test a single image on the trained model as follows

```
python tools/cityscapes/test_bisenetv2_cityscapes.py --weights_path ./weights/cityscapes/bisenetv2/cityscapes.ckpt 
--src_image_path ./data/test_image/cityscapes/test_01.png
```

If you want to evaluate the model on the whole cityscapes 
validation dataset you may call
```
python tools/cityscapes/evaluate_bisenetv2_cityscapes.py 
--pb_file_path ./checkpoint/bisenetv2_cityscapes_frozen.pb
--dataset_dir ./data/example_dataset/cityscapes
```

## Train model from scratch
#### Data Preparation
For speed up the training procedure. Convert the origin training
images into tensorflow records was highly recommended here which
is also very memory consuming. If you don't have enough ROM you 
may adjust the data_provider in training scripts into 
./data_provider/cityscapes_reader and use feed dict to train 
the model which can be pretty slow. If you have enough ROM then
you may convert the training images into tensorflow records.

First modified the ./config/cityscapes_bisenetv2.yml with right
dataset dir path

`Config file dataset field`
![Cityscapes_config_file_dataset_filed](./data/source_image/dataset_path.png)

Then sse the script here to generate the tensorflow records file

```
python tools/cityscapes/make_cityscapes_tfrecords.py
```

#### Train model
You may start your training procedure simply by

```
CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python tools/cityscapes/train_bisenetv2_cityscapes.py
```



## Acknowledgement
Mainly from [bisenetv2-tensorflow](https://github.com/MaybeShewill-CV/bisenetv2-tensorflow) 
