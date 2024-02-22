import numpy as np
import cv2
import os
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
import math
from PIL import Image


def find_freq_pixel(image):
        # resize image
        width, height = 150,150
        image = image.resize((width, height),resample = 0)
        
        # Medapatkan pixel atau warna image
        p = image.getcolors(width * height)
        
        # Sort berdasarkan jumlah ditemukannya
        sorted_p = sorted(p, key=lambda t: t[0])
        
        # Mengambil pixel dengan jumlah terbanyak
        freq_color = sorted_p[-1][1]
        
        return freq_color

def preprocess_img(img, imgSize):

    # Jika terdapat image yang rusak 
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]]) 
        print("Image not found")

    # Ambil size dan bentuk image
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),max(min(ht, int(h / f)), 1)) 
    
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC) 
    most_freq_pixel=find_freq_pixel(Image.fromarray(img))
    target = np.ones([ht, wt]) * most_freq_pixel  
    target[0:newSize[1], 0:newSize[0]] = img
    img = target

    return img

def pad_img(img):
    first_h,first_w=img.shape[0],img.shape[1]

    #perbesar/ pad ketinggian image

    if first_h<512:
        padding = np.ones((512-first_h,first_w))*255
        img = np.concatenate((img,padding))
        new_height=512
    else:
        padding = np.ones((roundup(first_h)-first_h,first_w))*255
        img = np.concatenate((img,padding))
        new_height = roundup(first_h)

    #perbesar/ pad kelebaran image
    if first_w<512:
        padding = np.ones((new_height,512-first_w))*255
        img = np.concatenate((img,padding),axis=1)
        new_width=512
    else:
        padding = np.ones((new_height,roundup(first_w)-first_w))*255
        img=np.concatenate((img,padding),axis=1)
        new_width = roundup(first_w)-first_w
        
    return img

def roundup(x):
    # Tinggikan nilai / roundup
    return int(math.ceil(x / 10.0)) * 10

# Unet
def unet(pretrained = None,input_size = (512,512,1)):
    inputs = Input(input_size)
    convo1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    convo1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convo1)
    mpool1 = MaxPooling2D(pool_size=(2, 2))(convo1)
    
    convo2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(mpool1)
    convo2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convo2)
    mpool2 = MaxPooling2D(pool_size=(2, 2))(convo2)
    
    convo3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(mpool2)
    convo3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convo3)
    mpool3 = MaxPooling2D(pool_size=(2, 2))(convo3)
    
    convo4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(mpool3)
    convo4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convo4)
    drop4 = Dropout(0.5)(convo4)
    mpool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    convo5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(mpool4)
    convo5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convo5)
    drop5 = Dropout(0.5)(convo5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    convo6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    convo6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convo6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(convo6))
    merge7 = concatenate([convo3,up7], axis = 3)
    convo7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    convo7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convo7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(convo7))
    merge8 = concatenate([convo2,up8], axis = 3)
    convo8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    convo8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convo8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(convo8))
    merge9 = concatenate([convo1,up9], axis = 3)
    convo9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    convo9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convo9)
    convo9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convo9)
    convo10 = Conv2D(1, 1, activation = 'sigmoid')(convo9)

    model = Model(inputs,convo10)

    model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if(pretrained):
      model.load_weights(pretrained)

    return model


model=unet()
model.load_weights('word_seg_model.h5')

def sort_word(wordlist):
    wordlist.sort(key=lambda x:x[0])
    return wordlist


def segment_into_words(line_img,idx):
    # Function ini akan mengambil gambar setiap line serta index line dan mengembalikan gambar setiap kata dan index line mereka
    img=pad_img(line_img)
    ori_img=img.copy()
    ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
    
    img=cv2.resize(img,(512,512))
    img=np.expand_dims(img,axis=-1)
    
    img=img/255
    img=np.expand_dims(img,axis=0)
    
    w_pred=model.predict(img)
    w_pred=np.squeeze(np.squeeze(w_pred,axis=0),axis=-1)
    w_pred=cv2.normalize(src=w_pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    cv2.threshold(w_pred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,w_pred)
    contours,_ = cv2.findContours(w_pred, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    (H, W) = ori_img.shape[:2]
    (newW, newH) = (512, 512)
    rW = W / float(newW)
    rH = H / float(newH)

    coor=[]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        coor.append((int(x*rW),int(y*rH),int((x+w)*rW),int((y+h)*rH)))
        
    # sort berdasarkan koordinasi contour.
    coor=sort_word(coor) 

    word_array=[]
    line_indicator=[]

    for (x1,y1,x2,y2) in coor:
        word_img=ori_img[y1:y2,x1:x2]
        word_img=preprocess_img(word_img,(128,32))
        word_img=np.expand_dims(word_img,axis=-1)
        word_array.append(word_img)
        line_indicator.append(idx)

    return line_indicator,word_array
    








