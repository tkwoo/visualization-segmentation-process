import keras
import cv2
import numpy as np

def train_visualization(model):
    imgInput = cv2.imread('./dataset/city/train/IMAGE/ori/img1.png', 0)
    input_data = imgInput.reshape((1,256,256,1))

    t_start = cv2.getTickCount()
    result = model.predict(input_data, 1)
    t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
    print "Predict Time: %.3f ms"%t_total

    imgMask = (result[0]*255).astype(np.uint8)
    imgShow = cv2.cvtColor(imgInput, cv2.COLOR_GRAY2BGR)
    # _, imgMask = cv2.threshold(imgMask, int(255*flag.confidence_value), 255, cv2.THRESH_BINARY)
    imgMaskColor = cv2.applyColorMap(imgMask, cv2.COLORMAP_JET)
    # imgZero = np.zeros((256,256), np.uint8)
    # imgMaskColor = cv2.merge((imgZero, imgMask, imgMask))
    imgShow = cv2.addWeighted(imgShow, 0.9, imgMaskColor, 0.3, 0.0)
    # output_path = os.path.join(flag.output_dir, os.path.basename(flag.test_image_path))
    # cv2.imwrite(output_path, imgShow)
    # print "SAVE:[%s]"%output_path
    cv2.namedWindow("show", 0)
    cv2.resizeWindow("show", 800, 800)
    cv2.imshow("show", imgShow)
    cv2.waitKey(1)

class trainCheck(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.aucs = []
		self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
            train_visualization(self.model)
		# self.losses.append(logs.get('loss'))
		# y_pred = self.model.predict(self.model.validation_data[0])
		# self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return
    