
import time
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')
# import nest_asyncio
# nest_asyncio.apply()

img_size = 128
classes = ['A', 'Ba', 'Da', 'Ga', 'Ha', 'Ja', 'Ka', 'La', 'Ma', 'Na', 'Pa', 'Ra', 'Ta', 'Tha', 'Wa', 'Ya']

loaded_model = load_model('model.h5')

"""## Test on Sample Image"""

plt.axis('off')

test_img = cv2.imread('LettersData/validation/A/A3 - Copy.png')

plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.show()

test_img1 = cv2.imread('LettersData/validation/Ba/Ba1 - Copy.png')

plt.imshow(cv2.cvtColor(test_img1, cv2.COLOR_BGR2RGB))
plt.show()

test_img2 = cv2.imread('LettersData/validation/Ga/Ga1.png')

plt.imshow(cv2.cvtColor(test_img2, cv2.COLOR_BGR2RGB))
plt.show()

test_img3 = cv2.imread('LettersData/validation/Ka/Ka0.png')

plt.imshow(cv2.cvtColor(test_img3, cv2.COLOR_BGR2RGB))
plt.show()

def sample_prediction(test_im):
    test_im = cv2.resize(test_im, (img_size, img_size), cv2.INTER_LINEAR) / 255
    test_pred = np.argmax(loaded_model(test_im.reshape((1, img_size, img_size, 3))))
    return classes[test_pred]

print("Predicted class for test_img: {}".format(sample_prediction(test_img)))
print("Predicted class for test_img1: {}".format(sample_prediction(test_img1)))
print("Predicted class for test_img2: {}".format(sample_prediction(test_img2)))
print("Predicted class for test_img3: {}".format(sample_prediction(test_img3)))