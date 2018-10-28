import sys
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import ImagePreprocessing

#CONSTANTS
NO_OF_CLASSES = 11
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
BATCH_SIZE = 30
DATASET_PATH = '../Sign-Language-Digits-Dataset/GrayScaleDataset/'
EPOCHS = 30

import keras
classifier= keras.models.load_model('new-mod20.model')

def predict(image_data):
    x = classifier.predict(image_data)
    x = x[0]
    maxi , maxv = 0 , x[0]
    for i in range(NO_OF_CLASSES):
        if(maxv < x[i]):
            maxv = x[i]
            maxi = i
            
    maxi = str(maxi)
    if(maxi == '10'):
        maxi = 'nothing'
    return (maxi , maxv)

c = 0

cap = cv2.VideoCapture(0)

res, score = '', 0.0
i = 0
mem = ''
consecutive = 0
sequence = ''
img_number = 1
A = []
flag = False
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    
    if ret:
        x1, y1, x2, y2 = 300, 100, 400, 200
        img_c = img[y1:y2, x1:x2]

        
        #img_g = convertToGreyScale(img_c)
        c += 1
#        image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
        
        a = cv2.waitKey(1) # waits to see if `esc` is pressed
        
        
        if i%10 == 0:
#            cv2.imwrite('saved/{i}test.jpg',img_c)
            cv2.imwrite('./LiveImages/image{}.jpg'.format(img_number) , img_c)
            img_c = cv2.imread('./LiveImages/image{}.jpg'.format(img_number) , 0)
            cv2.imwrite('./LiveImages/image-gray{}.jpg'.format(img_number) , img_c)
            img_c = cv2.imread('./LiveImages/image-gray{}.jpg'.format(img_number))
            img_c = cv2.resize(img_c,(IMAGE_HEIGHT , IMAGE_WIDTH))
            img_c = img_c.reshape(1 , IMAGE_HEIGHT , IMAGE_WIDTH , 3)
            
            img_number+= 1
            res_tmp, score = predict(img_c)
            res = res_tmp
            if res == 'nothing':
                flag = True
                
            if mem == res:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive == 2 and res not in ['nothing']:
                if res == 'space':
                    sequence += ' '
                elif res == 'del':
                    sequence = sequence[:-1]
                else:
                    if flag:    
                        sequence += res
                        flag = False
                consecutive = 0
        i += 1
        cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
        cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        mem = res
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.imshow("img", img)
        img_sequence = np.zeros((200,1200,3), np.uint8)
        cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('sequence', img_sequence)
        
        if a == 27: # when `esc` is pressed
            break

# Following line should... <-- This should work fine now
cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()