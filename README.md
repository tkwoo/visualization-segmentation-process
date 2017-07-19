# Segmentation using Unet open version

Semenatic segmentation using Unet  

## Result

[![Youtube video](https://youtu.be/GYoyzB7aoK4/0.jpg)](https://youtu.be/GYoyzB7aoK4?t=24)
  
## Requirements

- Python 2.7
- OpenCV 3.2.0
- [Keras 2.0.5](https://github.com/fchollet/keras)
- [TensorFlow 1.2.0](https://github.com/tensorflow/tensorflow)

## Usage  

Input data(only for training)

    └── dataset
        └── xxx
            └── train
                └── IMAGE
                    └── ori
                        └── xxx.png (name doesn't matter)
                └── GT
                    └── mask
                        └── xxx.png (It must have same name as original image)

The dataset directory structure is quite complex to use the Keras DataGen Framework.

Input data for testing

    └── test_data
        └── SMC_ku_001
            └── T1.mgz
        
First, create checkpoint dir and download trained parameter files  

    └── checkpoint
        └── (ckpt_name)
            ├── model.json 
            ├── weight.xx.h5
            └── ...

You can download **CHECKPOINT** files in project
  
To test a model

    $ python main.py --mode predict_mri --ckpt_name Unet_sagitalBM --T1_path ./test_data/SMC_ku_001/T1.mgz --subject_name SMC001 --debug false --output_dir ./result

To test a model

    $ python main.py --mode train --ckpt_name <NAME> --data_path <./dataset/...>


### Reference
paper : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/  

