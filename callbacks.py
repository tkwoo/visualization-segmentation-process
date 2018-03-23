from __future__ import print_function
import keras
import cv2
import numpy as np
import os
from glob import glob

class trainCheck(keras.callbacks.Callback):
    def __init__(self, flag):
        self.flag = flag
        self.epoch = 0

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        return

    def on_epoch_end(self, epoch, logs={}):
        self.train_visualization_seg(self.model, epoch)

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    def train_visualization_seg(self, model, epoch):
        image_name_list = sorted(glob(os.path.join(self.flag.data_path,'train/IMAGE/*/*.png')))

        image_name = image_name_list[-1]
        height = self.flag.image_height
        width = self.flag.image_width
        
        imgInput = cv2.imread(image_name, self.flag.color_mode)
        imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)

        output_path = self.flag.output_dir
        input_data = imgInput.reshape((1, height, width, self.flag.color_mode*2+1))

        t_start = cv2.getTickCount()
        result = model.predict(input_data, 1)
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
        print ("[*] %s predict time: %.3f ms"%(os.path.basename(image_name),t_total))

        imgMask = (result[0]*255).astype(np.uint8)
        imgShow = cv2.cvtColor(imgInput, cv2.COLOR_RGB2BGR).copy()
        imgMaskColor = imgMask
        imgShow = cv2.addWeighted(imgShow, 0.5, imgMaskColor, 0.6, 0.0)
        output_path = os.path.join(self.flag.output_dir, '%04d_'%epoch+os.path.basename(image_name))
        mask_path = os.path.join(self.flag.output_dir, 'mask_%04d_'%epoch+os.path.basename(image_name))
        cv2.imwrite(output_path, imgShow)
        cv2.imwrite(mask_path, imgMaskColor)
        # print "SAVE:[%s]"%output_path
        # cv2.imwrite(os.path.join(output_path, 'img%04d.png'%epoch), imgShow)
        # cv2.namedWindow("show", 0)
        # cv2.resizeWindow("show", 800, 800)
        # cv2.imshow("show", imgShow)
        # cv2.waitKey(1)