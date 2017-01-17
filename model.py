import csv
import os
import glob
import numpy as np
import cv2
import random
from PIL import Image
from PIL import ImageOps
#import argparse
#import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

IMG_RES = 64
IMG_X = IMG_RES
IMG_Y = IMG_RES
BATCHSIZE = 64
DATASETS = ['udacity', 'track1_smooth', 'track1_recoverLeft', 'track1_recoverRight']
EPOCHS = 10
#MODEL_PATH = 'F:/GitHub/CarND-Behavioral-Cloning-P3/models/'
MODEL_PATH = 'C:/Users/Marcus/Documents/GitHub/CarND-Behavioral-Cloning-P3/models/'
DEBUG = False

def readCSV(dataSet, keep=1.0, keepStraight=0.5, useStereo=False, flip=False):
    '''
    Reads data from drive log CSV and generates array of training data {img path, steering angle, img needs flipping}
    '''
    stereoSteerOffset = 0.25
    dataDir = 'driveData/'
    path = '%s%s/' % (dataDir, dataSet)
    # read in csv
    # centerImg, leftImg, rightImg, steering angle, throttle, break, speed
    csvFileName = '%s/driving_log.csv' % (path)
    csvData = []
    with open(csvFileName, 'rt') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        for row in reader:
            csvData.append( {'center': '%s%s' % (path, row[0]), 'left': '%s%s' % (path, row[1]), 'right': '%s%s' % (path, row[2]), 'steer': float(row[3])})
    data = []
    for line in csvData:
        randKeep = np.random.uniform(0,1)
        if randKeep < keep:
            randKeepStraight = np.random.uniform(0,1)
            if (abs(line['steer']) >= 0.05) or ((abs(line['steer']) < 0.05) and (randKeepStraight < keepStraight)):
                data.append({'img': line['center'], 'steer': line['steer'], 'flip': False})
                if flip:
                    data.append({'img': line['center'], 'steer':line['steer']*-1, 'flip': True})
                if useStereo:
                    data.append({'img': line['left'], 'steer':line['steer']+stereoSteerOffset, 'flip': False})
                    data.append({'img': line['right'], 'steer':line['steer']-stereoSteerOffset, 'flip': False})
                    if flip:
                        data.append({'img': line['left'], 'steer':(line['steer']+stereoSteerOffset)*-1,'flip': True})
                        data.append({'img': line['right'], 'steer':(line['steer']-stereoSteerOffset)*-1 ,'flip': True})
    return data


def addShadow(img):
    brightness = 0.5
    height, width = img.shape[:2]
    shdwWidth = 20+width*np.random.uniform()
    shdwHeight = 20+height*np.random.uniform()

    center_x = width*np.random.uniform()
    center_y = height*np.random.uniform()

    left_x = center_x-(shdwWidth/2)
    right_x = center_x+(shdwWidth/2)
    top_y = center_y-(shdwHeight/2)
    bot_y = center_y+(shdwHeight/2)
    
    # ugly bounding box test...
    for x in range(len(img)):
        for y in range(len(img[x])):
            if (x < right_x) and (x > left_x) and (y > top_y) and (y < bot_y):
                img[x][y] = img[x][y] * brightness
    return img


def randBrightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    randBright = min(0.25+np.random.uniform(), 1.0)
    hsv[:,:,2] = hsv[:,:,2] * randBright
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img


def preprocessImg(img, cropYtop=35, cropYbot=25, targetResize=IMG_RES, blur=False, flip=False, addShdw=False, randBright=False):
    # convert from PIL to cv2
    img = np.array(img)
    height, width = img.shape[:2]
    # CROP
    imgCrop = img[cropYtop:height-cropYbot, 0:width]
    # RANDOMIZE BRIGHTNESS
    if randBright:
        imgCrop = randBrightness(imgCrop)
    # ADD SHADOW
    if addShdw:
        imgCrop = addShadow(imgCrop)
    # RESIZE and make square (that way I can crop however I want and not worry about my model not fitting anymore)
    imgRes = cv2.resize(imgCrop, (targetResize, targetResize))
    # BLUR
    if blur:
        imgProc = cv2.GaussianBlur(imgRes,(3,3),0)
    else:
        imgProc = imgRes
    # FLIP
    if flip:
        imgProc = np.fliplr(imgProc)
    return imgProc


def generateBatchRandom(trainData):
    batchImg = np.zeros((BATCHSIZE, IMG_Y, IMG_X, 3))
    batchSteer = np.zeros(BATCHSIZE)
    while 1:
        for i in range(BATCHSIZE):
            i_data = np.random.randint(len(trainData))
            img = Image.open(trainData[i_data]['img'])
            batchImg[i] = preprocessImg(img, flip=trainData[i_data]['flip'], addShdw=True, randBright=True)
            batchSteer[i] = trainData[i_data]['steer']
        yield batchImg, batchSteer


def generateBatch(trainData):
    batchImg = np.zeros((BATCHSIZE, IMG_Y, IMG_X, 3))
    batchSteer = np.zeros(BATCHSIZE)
    while 1:
        totalCount = 1
        batchIndex = 0
        for data in trainData:
            img = Image.open(data['img'])                   
            batchImg[batchIndex] = preprocessImg(img, flip=data['flip'])
            batchSteer[batchIndex] = data['steer']
            if (batchIndex == BATCHSIZE-1) or (totalCount == len(trainData)):
                if DEBUG:
                    print('\nyielding batch up to %s with batch size %s' % (totalCount, batchIndex+1))
                # cut down batch array if it is smaller than BATCHSIZE (otherwise we pass zero-values to the trainer)
                if batchIndex < BATCHSIZE-1:
                    batchImg = batchImg[:batchIndex+1]
                    batchSteer = batchSteer[:batchIndex+1]
                yield batchImg, batchSteer
                # reset everything
                batchImg = np.zeros((BATCHSIZE, IMG_Y, IMG_X, 3))
                batchSteer = np.zeros(BATCHSIZE)
                batchIndex = -1
            batchIndex += 1
            totalCount += 1


def nVidia(ch, row, col):
    #ch, row, col = 3, 160, 320  # original model format
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(row, col, ch),
              output_shape=(row, col, ch)))
    #model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    #model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Convolution2D(36, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    #model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Convolution2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(ELU())
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    return model

def commaAI(ch, row, col):
    #ch, row, col = 3, 160, 320  # original model format
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(row, col, ch),
              output_shape=(row, col, ch)))
    #model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(Convolution2D(16, 5, 5, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    #model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    return model

def dummyModel(ch, row, col):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(row, col, ch),
              output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    print("reading csv...")
    trainData = []
    for dataset in DATASETS:
        trainData.extend(readCSV(dataset, useStereo=True, flip=True, keepStraight=0.5, keep = 1.0))
    print("total samples: %s" % len(trainData))
    
    for i in range(2):
        print("building model...")
        #model = nVidia(3, IMG_Y, IMG_X)
        model = commaAI(3, IMG_Y, IMG_X)

        print("starting training...")
        generator = generateBatchRandom(trainData)

        model.fit_generator(generator, samples_per_epoch=20000, nb_epoch=EPOCHS)

        print("saving model...")
        modelName = 'commaAI_bigData_20k_%se_%s' % (EPOCHS, i)
        fileName = '%s%s' % (MODEL_PATH, modelName)
        json_string = model.to_json()
        with open('%s.json' % fileName, "w") as json_file:
            json_file.write(json_string)
        model.save_weights('%s.h5' % fileName)

if __name__ == '__main__':
    main()