#imports
import pandas as pd
import os
import ast

import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import matplotlib.pyplot as plt
import numpy as np

import pylab

import librosa

import ffmpeg
import audioread
import sklearn
import librosa.display
import datetime
import time

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Bidirectional, LSTM, Activation, GRU, Conv2D, concatenate, MaxPooling2D, Flatten, Embedding, Lambda, Reshape

from keras.optimizers import Adam, RMSprop
from keras import backend as K

#plot_file(path)
#function to plot spectrograms
def plot_file(path):
    currSpect=path
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(currSpect.T,x_axis = 'time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    print(currSpect.shape)



#Load and trim datasets to shread out useless info
filePath = 'D:\\fma_metadata\\tracks.csv'
df_tracks = pd.read_csv(filePath, index_col=0, header=[0, 1])
print(list(df_tracks))
filter = [('set', 'split'), ('set', 'subset') , ('track', 'genre_top')]
df_sel = df_tracks[filter]
df_sel = df_sel[df_sel[filter[1]]=='small']
df_sel['track_id'] = df_sel.index
df_test = df_sel[df_sel[filter[0]]=='test']
df_valid = df_sel[df_sel[filter[0]]=='validation']
df_train = df_sel[df_sel[filter[0]]=='training']

print(df_sel.tail())
print(df_test.shape)
print(df_test.head())
print( df_train.shape)
print(df_train.head())
print(df_valid.shape)
print(df_valid.head())
print(df_sel[filter[2]].value_counts())



#Build and train the model
def build_and_train_model():
    xTrain = np.load("D:\\spectAr2\\final_train.npy")
    yTrain = np.load("D:\\spectAr2\\genres_train.npy")

    print(yTrain.shape)

    xValid = np.load("D:\\spectAr2\\final_valid.npy")
    yValid = np.load("D:\\spectAr2\\genres_valid.npy")

    xTest = np.load("D:\\spectAr2\\final_test.npy")
    yTest = np.load("D:\\spectAr2\\genres_test.npy")


    yTrain=keras.utils.np_utils.to_categorical(yTrain)
    yValid=keras.utils.np_utils.to_categorical(yValid)
    yTest=keras.utils.np_utils.to_categorical(yTest)
    
    #-----------------------------------------------------------------------------------
    #  base sequential model
    #-----------------------------------------------------------------------------------

    # print("start.")
    # model = Sequential()
    # model.add(Conv2D(filters=16, input_shape=(128,640,1), kernel_size=(1,3), padding='valid', activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))
    # model.add(Conv2D(filters=32, kernel_size=(1,3), padding='valid', activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))
    # model.add(Conv2D(filters=64, kernel_size=(1,3), padding='valid', activation='relu'))
    # model.add(MaxPooling2D((2,2),strides=(2,2)))
    # model.add(Conv2D(filters=128, kernel_size=(1,3), padding='valid', activation='relu'))
    # model.add(MaxPooling2D((4,4), strides=(4,4)))
    # model.add(Conv2D(filters=64, kernel_size=(1,3), padding='valid', activation='relu'))
    # model.add(MaxPooling2D((4,4), strides=(4,4)))
    # model.add(Flatten())
    # model.add( Dense(8, activation = 'softmax', name='preds'))
    
    
    #-------------------------------------------------------------------------------------
    # parallel CNN/RNN model
    #-------------------------------------------------------------------------------------

    inp = Input((128,640,1))
    c1 = Conv2D(filters=16, kernel_size=(1,3), padding='valid', activation='relu')(inp)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(c1)
    c2 = Conv2D(filters=32, kernel_size=(1,3), padding='valid', activation='relu')(pool1)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(c2)
    c3 = Conv2D(filters=64, kernel_size=(1,3), padding='valid', activation='relu')(pool2)
    pool3 = MaxPooling2D((2,2),strides=(2,2))(c3)
    c4 = Conv2D(filters=64, kernel_size=(1,3), padding='valid', activation='relu')(pool3)
    pool4 = MaxPooling2D((4,4), strides=(4,4))(c4)
    c5 = Conv2D(filters=64, kernel_size=(1,3), padding='valid', activation='relu')(pool4)
    pool5 = MaxPooling2D((4,4), strides=(4,4))(c5)
    
    flat1 = Flatten()(pool5)
    
    # GRU block
    
    GRUPool = MaxPooling2D((2,4), strides = (2,4))(inp)
    squeezed = Lambda(lambda x: K.squeeze(x, axis= -1))(lstmPool)
    lstm = Bidirectional(GRU(64))(squeezed)
    concatenated = concatenate([flat1, lstm], axis = 1)
    
    output = Dense(8, activation='softmax')(concatenated)
    
    model = Model(inp, output)

    
    optimizer = RMSprop(lr=0.0005)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary)

    result = model.fit(xTrain, yTrain, batch_size=64, epochs=50, validation_data=(xValid,yValid), verbose=1)
    score = model.evaluate(xTest, yTest, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])




def make_filename(num):
    result = "%i" % (num)
    while len(result)<6:
        result = "0"+result

    result = result+".npy"

    return result



#creates training, testing and validation datasets.
def create_separate_datasets():
    genres_train = []
    genres_valid = []
    genres_test = []

    for roots, dirs, files in os.walk("D:\\spectrograms2"):
        for index, rows in df_train.iterrows():
            temp = make_filename(index)
            if temp in files:
                genre = str(rows[('track', 'genre_top')])
                if(genre == 'International'):
                    genres_train.append(0)
                elif(genre == 'Electronic'):
                    genres_train.append(1)
                elif(genre=='Hip-Hop'):
                    genres_train.append(2)
                elif (genre == 'Folk'):
                    genres_train.append(3)
                elif (genre == 'Instrumental'):
                    genres_train.append(4)
                elif (genre == 'Pop'):
                    genres_train.append(5)
                elif (genre == 'Experimental'):
                    genres_train.append(6)
                elif (genre == 'Rock'):
                    genres_train.append(7)

    for index, rows in df_valid.iterrows():
        genre = str(rows[('track', 'genre_top')])
        if(genre == 'International'):
            genres_valid.append(0)
        elif(genre == 'Electronic'):
            genres_valid.append(1)
        elif(genre=='Hip-Hop'):
            genres_valid.append(2)
        elif (genre == 'Folk'):
            genres_valid.append(3)
        elif (genre == 'Instrumental'):
            genres_valid.append(4)
        elif (genre == 'Pop'):
            genres_valid.append(5)
        elif (genre == 'Experimental'):
            genres_valid.append(6)
        elif (genre == 'Rock'):
            genres_valid.append(7)

    for index, rows in df_test.iterrows():
        genre = str(rows[('track', 'genre_top')])
        if (genre == 'International'):
            genres_test.append(0)
        elif (genre == 'Electronic'):
            genres_test.append(1)
        elif (genre == 'Hip-Hop'):
            genres_test.append(2)
        elif (genre == 'Folk'):
            genres_test.append(3)
        elif (genre == 'Instrumental'):
            genres_test.append(4)
        elif (genre == 'Pop'):
            genres_test.append(5)
        elif (genre == 'Experimental'):
            genres_test.append(6)
        elif (genre == 'Rock'):
            genres_test.append(7)

    genres_test=np.array(genres_test)
    genres_train=np.array(genres_train)
    genres_valid=np.array(genres_valid)

    print(genres_train.shape)

    np.save("D:\\spectAr2\\genres_valid.npy", genres_valid)
    np.save("D:\\spectAr2\\genres_test.npy", genres_test)
    np.save("D:\\spectAr2\\genres_train.npy", genres_train)

    print(genres_valid)
    print(genres_train)
    print(genres_test)


    spect_train = np.empty((0,128,640))
    spect_test = np.empty((0,128,640))
    spect_valid = np.empty((0,128,640))

    print("start")
    start_time = time.time()
    foo = 1
    count = 1
    num = 1
    for roots, dirs, files in os.walk("D:\\spectrograms2"):
        for file in files:
            # if(num>5):
            #     break
            try:
                if(count> 5):
                    count = 1

                    spect_train = np.expand_dims(spect_train, axis=3)
                    spect_test = np.expand_dims(spect_test, axis=3)
                    spect_valid = np.expand_dims(spect_valid, axis=3)

                    np.save("D:\\spectAr2\\spect_train" + str(num) + ".npy", spect_train)
                    np.save("D:\\spectAr2\\spect_test" + str(num) + ".npy", spect_test)
                    np.save("D:\\spectAr2\\spect_valid" + str(num) + ".npy", spect_valid)

                    print("Curr is: " + str(num))
                    print(spect_train.shape)
                    print(spect_test.shape)
                    print(spect_valid.shape)

                    spect_train = np.empty((0, 128, 640))
                    spect_test = np.empty((0, 128, 640))
                    spect_valid = np.empty((0, 128, 640))
                    num+=1
                # print(file)
                f=np.load("D:\\spectrograms2\\"+ file)
                # plot_file(f)
                curr = int(file[:6])
                if curr in df_train.index:
                    spect_train = np.append(spect_train, [f], axis=0)
                elif curr in df_test.index:
                    spect_test = np.append(spect_test, [f], axis=0)
                elif curr in df_valid.index:
                    spect_valid = np.append(spect_valid, [f], axis=0)
                foo+=1

                if (foo > 100):
                    foo=1
                    count += 1
                    print(count)
            except:
                print("Couldn't process: "+ file)

    print(count)

    spect_train = np.expand_dims(spect_train, axis=3)
    spect_test = np.expand_dims(spect_test, axis=3)
    spect_valid = np.expand_dims(spect_valid, axis=3)

    np.save("D:\\spectAr2\\spect_train" + str(num) + ".npy", spect_train)
    np.save("D:\\spectAr2\\spect_test" + str(num) + ".npy", spect_test)
    np.save("D:\\spectAr2\\spect_valid" + str(num) + ".npy", spect_valid)

    print("Curr is: " + str(num))
    print(spect_train.shape)
    print(spect_test.shape)
    print(spect_valid.shape)
    print("end")
    print(time.time()-start_time)
    print(spect_train.shape)
    print(spect_test.shape)
    print(spect_valid.shape)

#concatenates fragmented datasets.
def concatenate_datasets():
    at=np.load("D:\\spectAr2\\spect_test1.npy")
    bt=np.load("D:\\spectAr2\\spect_test2.npy")
    ct=np.load("D:\\spectAr2\\spect_test3.npy")
    dt=np.load("D:\\spectAr2\\spect_test4.npy")
    et=np.load("D:\\spectAr2\\spect_test5.npy")
    ft=np.load("D:\\spectAr2\\spect_test6.npy")
    gt=np.load("D:\\spectAr2\\spect_test7.npy")
    ht=np.load("D:\\spectAr2\\spect_test8.npy")
    it=np.load("D:\\spectAr2\\spect_test9.npy")
    jt=np.load("D:\\spectAr2\\spect_test10.npy")
    kt=np.load("D:\\spectAr2\\spect_test11.npy")
    lt=np.load("D:\\spectAr2\\spect_test12.npy")
    mt=np.load("D:\\spectAr2\\spect_test13.npy")
    nt=np.load("D:\\spectAr2\\spect_test14.npy")
    ot=np.load("D:\\spectAr2\\spect_test15.npy")
    pt=np.load("D:\\spectAr2\\spect_test16.npy")

    atr=np.load("D:\\spectAr2\\spect_train1.npy")
    btr=np.load("D:\\spectAr2\\spect_train2.npy")
    ctr=np.load("D:\\spectAr2\\spect_train3.npy")
    dtr=np.load("D:\\spectAr2\\spect_train4.npy")
    etr=np.load("D:\\spectAr2\\spect_train5.npy")
    ftr=np.load("D:\\spectAr2\\spect_train6.npy")
    gtr=np.load("D:\\spectAr2\\spect_train7.npy")
    htr=np.load("D:\\spectAr2\\spect_train8.npy")
    itr=np.load("D:\\spectAr2\\spect_train9.npy")
    jtr=np.load("D:\\spectAr2\\spect_train10.npy")
    ktr=np.load("D:\\spectAr2\\spect_train11.npy")
    ltr=np.load("D:\\spectAr2\\spect_train12.npy")
    mtr=np.load("D:\\spectAr2\\spect_train13.npy")
    ntr=np.load("D:\\spectAr2\\spect_train14.npy")
    otr=np.load("D:\\spectAr2\\spect_train15.npy")
    ptr=np.load("D:\\spectAr2\\spect_train16.npy")

    av=np.load("D:\\spectAr2\\spect_valid1.npy")
    bv=np.load("D:\\spectAr2\\spect_valid2.npy")
    cv=np.load("D:\\spectAr2\\spect_valid3.npy")
    dv=np.load("D:\\spectAr2\\spect_valid4.npy")
    ev=np.load("D:\\spectAr2\\spect_valid5.npy")
    fv=np.load("D:\\spectAr2\\spect_valid6.npy")
    gv=np.load("D:\\spectAr2\\spect_valid7.npy")
    hv=np.load("D:\\spectAr2\\spect_valid8.npy")
    iv=np.load("D:\\spectAr2\\spect_valid9.npy")
    jv=np.load("D:\\spectAr2\\spect_valid10.npy")
    kv=np.load("D:\\spectAr2\\spect_valid11.npy")
    lv=np.load("D:\\spectAr2\\spect_valid12.npy")
    mv=np.load("D:\\spectAr2\\spect_valid13.npy")
    nv=np.load("D:\\spectAr2\\spect_valid14.npy")
    ov=np.load("D:\\spectAr2\\spect_valid15.npy")
    pv=np.load("D:\\spectAr2\\spect_valid16.npy")

    print(df_train.shape)
    result1 = np.concatenate((at,bt,ct,dt,et,ft,gt,ht,it,jt,kt,lt,mt,nt,ot,pt), axis=0)
    print("YAY!")
    result2 = np.concatenate((atr,btr,ctr,dtr,etr,ftr,gtr,htr,itr,jtr,ktr,ltr,mtr,ntr,otr,ptr), axis=0)
    result3 = np.concatenate((av,bv,cv,dv,ev,fv,gv,hv,iv,jv,kv,lv,mv,nv,ov,pv), axis=0)

    np.save("D:\\spectAr2\\final_test.npy", result1)
    np.save("D:\\spectAr2\\final_valid.npy", result3)
    np.save("D:\\spectAr2\\final_train.npy", result2)
    print(result1.shape)
    print(result2.shape)
    print(result3.shape)




# concatenate_datasets()  #concatinate fragmented datasets
build_and_train_model()
# create_separate_datasets()    #create training, testing and validation datasets





