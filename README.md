# Visualization segmentation training process 

Semenatic segmentation using Unet, fcn, pspnet

## Result

[![Youtube video](./result.gif)](https://youtu.be/TtFAXYQc54o)

click image to watch video


## Requirements

- Python 2.7 or 3.6
- OpenCV 3.4.0
- [Keras 2.1.4](https://github.com/fchollet/keras)
- [TensorFlow 1.5.0](https://github.com/tensorflow/tensorflow)

## Usage  

To train a model (**visualization**)

    $ python main.py

Then, the training steps(image) will be saved 'result' directory

-----------

    usage: main.py [-h] [--data_path DATA_PATH] 
                        [--output_dir OUTPUT_DIR]
                        [--image_height IMAGE_HEIGHT] 
                        [--image_width IMAGE_WIDTH]
                        [--batch_size BATCH_SIZE]
                        [--total_epoch TOTAL_EPOCH]
                        [--initial_learning_rate INITIAL_LEARNING_RATE]
                        [--learning_rate_decay_factor LEARNING_RATE_DECAY_FACTOR]
                        [--epoch_per_decay EPOCH_PER_DECAY] 
                        [--ckpt_dir CKPT_DIR]
                        [--ckpt_name CKPT_NAME]
                        [--pretrained_weight_path PRETRAINED_WEIGHT_PATH]
                        [--confidence_value CONFIDENCE_VALUE] 
                        [--debug DEBUG]
                        [--mode MODE] 
                        [--test_image_path TEST_IMAGE_PATH]
                        [--tf_log_level TF_LOG_LEVEL]

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

The dataset directory structure is quite complex to use the Keras *ImageDataGenerator* Framework.

Input data for testing

    └── test_data
        └── image.png
        
First, create checkpoint dir and download trained parameter files  

    └── checkpoint
        └── (ckpt_name)
            ├── model.json 
            ├── weight.xx.h5
            └── ...

You can download **CHECKPOINT** --> not supported



To test a model

    $ python main.py --mode predict_img --ckpt_name <NAME> --test_image_path <.../image.png>


### Reference
paper : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/  

