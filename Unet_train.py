from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import keras
import cv2
import numpy as np
import os
from glob import glob
import argparse
import random
import math

from Unet_keras import get_unet
import callbacks

class TrainModel:
    def __init__(self, flag):
        self.flag = flag

    def train_generator(self, image_generator, mask_generator):
        while True:
            yield(next(image_generator), next(mask_generator))

    def lr_step_decay(self, epoch):
        init_lr = self.flag.initial_learning_rate
        lr_decay = self.flag.learning_rate_decay_factor
        epoch_per_decay = self.flag.epoch_per_decay
        lrate = init_lr * math.pow(lr_decay, math.floor((1+epoch)/epoch_per_decay))
        # print lrate
        return lrate

    def train_unet(self):

        img_size = self.flag.image_size
        batch_size = self.flag.batch_size
        epochs = self.flag.total_epoch

        datagen_args = dict(featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
                # fill_mode='constant',
                # cval=0.,
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images

        image_datagen = ImageDataGenerator(**datagen_args)
        mask_datagen = ImageDataGenerator(**datagen_args)

        seed = random.randrange(1, 1000)
        image_generator = image_datagen.flow_from_directory(
                    os.path.join(self.flag.data_path, 'train/IMAGE'),
                    class_mode=None, seed=seed, batch_size=batch_size, color_mode='grayscale')
        mask_generator = mask_datagen.flow_from_directory(
                    os.path.join(self.flag.data_path, 'train/GT'),
                    class_mode=None, seed=seed, batch_size=batch_size, color_mode='grayscale')
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        model = get_unet(self.flag)
        if self.flag.pretrained_weight_path != None:
            model.load_weights(self.flag.pretrained_weight_path)
        
        if not os.path.exists(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name)):
            os.mkdir(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name))
        model_json = model.to_json()
        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'w') as json_file:
            json_file.write(model_json)
        vis = callbacks.trainCheck()
        model_checkpoint = ModelCheckpoint(
                    os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name,'weights.{epoch:02d}.h5'), 
                    period=1000)
        learning_rate = LearningRateScheduler(self.lr_step_decay)
        model.fit_generator(
            self.train_generator(image_generator, mask_generator),
            steps_per_epoch= image_generator.n // batch_size,
            epochs=epochs,
            callbacks=[model_checkpoint, learning_rate, vis]
        )

