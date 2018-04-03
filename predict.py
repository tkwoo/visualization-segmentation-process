from __future__ import print_function
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_json
import tensorflow as tf
import keras
import cv2
import numpy as np
import os
from glob import glob
import argparse

def predict_image(flag):
    t_start = cv2.getTickCount()
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    with open(os.path.join(flag.ckpt_dir, flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    weight_list = sorted(glob(os.path.join(flag.ckpt_dir, flag.ckpt_name, "weight*")))
    model.load_weights(weight_list[-1])
    print ("[*] model load : %s"%weight_list[-1])
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
    print ("[*] model loading Time: %.3f ms"%t_total)

    bgr_img = cv2.imread(flag.test_image_path, 1)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    input_data = rgb_img[None,:,:,:]

    t_start = cv2.getTickCount()
    result = model.predict(input_data, 1)
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    print ("Predict Time: %.3f ms"%t_total)
    
    imgMask = (result[0]*255).astype(np.uint8)
    imgShow = bgr_img.copy()
    imgMaskColor = imgMask
    imgShow = cv2.addWeighted(imgShow, 0.5, imgMaskColor, 0.6, 0.0)
    output_path = os.path.join('./', os.path.basename(flag.test_image_path))
    cv2.imwrite(output_path, imgShow)
    print ("SAVE:[%s]"%output_path)
        

    