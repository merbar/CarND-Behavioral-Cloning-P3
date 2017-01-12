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
DATASETS = ['udacity']
EPOCHS = 1
MODEL_PATH = 'F:/GitHub/CarND-Behavioral-Cloning-P3/models/'
MODEL_JSON_FILE = 'F:/GitHub/CarND-Behavioral-Cloning-P3/models/test.json'
#MODEL_WEIGHT_FILE = 'C:/Users/Marcus/Documents/GitHub/CarND-Behavioral-Cloning-P3/models/test.h5'
#MODEL_JSON_FILE = 'C:/Users/Marcus/Documents/GitHub/CarND-Behavioral-Cloning-P3/models/test.json'
DEBUG = True

def readCSV(dataSet, keep=1.0, keepStraight=0.5, useStereo=False, flip=False):
    '''
    Reads data from drive log CSV and generates array of training data {img path, steering angle, img needs flipping}
    '''
    stereoSteerOffset = 0.1
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


def preprocessImg(img, cropYtop=60, cropYbot=25, targetResize=IMG_RES, blur=False, flip=False):
    # convert from PIL to cv2
    img = np.array(img) 
    height, width = img.shape[:2]
    # CROP
    imgCrop = img[cropYbot:height-cropYbot, 0:width]
    # RESIZE and make square (that way I can crop however I want and not worry about my model not fitting anymore)
    imgRes = cv2.resize(imgCrop, (targetResize, targetResize))
    # BLUR
    if blur:
        imgProc = cv2.GaussianBlur(imgRes,(3,3),0)
    else:
        imgProc = imgRes
    # NORMALIZE
    #imgProc = cv2.normalize(imgProc, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst = imgProc)
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
            batchImg[i] = preprocessImg(img, flip=trainData[i_data]['flip'])
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

            '''
            if DEBUG:
                if totalCount == 1:
                    #imgDebug = np.array(img)
                    imgDebug = preprocessImg(img, flip=data['flip'])
            '''
                    
            batchImg[batchIndex] = preprocessImg(img, flip=data['flip'])
            batchSteer[batchIndex] = data['steer']

            '''
            if DEBUG:
                if totalCount == 1:
                    print(batchImg[batchIndex])
                    #print('\n\n%s\n\n' % batchImg.shape)
            '''

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
    '''
    input (66x200)
    normalize
    conv 5x5 kernel, 2x2 stride (kernel might be too much with my smaller images)
    => 31x98x *24*
    conv 5x5, 2x2
    => 14x47x *36*
    conv 5x5, 2x2
    => 5x22x *48*
    conv 3x3, non strided
    => 3x20x *64*
    conv 3x3, non strided
    => 1x18x *64*
    flatten
    FC 1164
    FC 100
    FC 50
    FC 10
    '''
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

    '''
    img = Image.open(trainData[0]['img'])
    imgDebug = preprocessImg(img)
    cv2.imshow('', imgDebug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(imgDebug.shape)
    print(imgDebug)

    batchImg = np.zeros((1, IMG_Y, IMG_X, 3))
    print(batchImg[0][1][1])
    batchImg[0] = imgDebug
    cv2.imshow('', batchImg[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(batchImg[0].shape)
    print(batchImg[0])

    '''
    print("randomizing data (%s samples)..." % len(trainData))
    #random.shuffle(trainData)

    print("building model...")
    model = nVidia(3, IMG_Y, IMG_X)

    print("starting training...")
    #generator = generateBatch(trainData)
    #model.fit_generator(generator, samples_per_epoch=len(trainData), nb_epoch=EPOCHS)
    generator = generateBatchRandom(trainData)
    model.fit_generator(generator, samples_per_epoch=20000, nb_epoch=EPOCHS)

    print("saving model...")
    fileName = '%stest' % MODEL_PATH
    json_string = model.to_json()
    with open('%s.json' % fileName, "w") as json_file:
        json_file.write(json_string)
    model.save_weights('%s.h5' % fileName)
    
    

if __name__ == '__main__':
    main()