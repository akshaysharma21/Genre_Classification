import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import datetime

def plot_file(path):
    currSpect=path
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(currSpect.T,x_axis = 'time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    print(currSpect.shape)

#
# Code block for converting audio files spectrograms 
#
#--------------------------------------------------------------------------------------------------
# making the spectrogram dataset
#
#


foo = 0
paths = []
for roots, dirs, files in os.walk("D:\\fma_small"):
    for dir in dirs:
        for subRoots, subDirs, subFiles in os.walk("D:\\fma_small\\"+ dir):
            for file in subFiles:
                if(file[6:]==".mp3"):
                    paths.append("D:\\fma_small\\"+ dir +"\\" + file)
                    # foo+=1



print(paths[:8])
print(len(paths))

spectrograms = []

print(datetime.datetime.now().time())
for path in paths:

    y, sr = librosa.load(path)

    # print(y)

    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    spect = spect[:, :640]
    dest = "D:\\spectrograms\\"+ path[-10:-4]+".npy"
    np.save(dest, spect)

    spectrograms.append(spect)

print(datetime.datetime.now().time())

print(len(spectrograms))

for sp in spectrograms:
    print(sp.shape)


path = input("Enter the full path of the file to be plotted: ")

while path[-4:]== ".npy":
    plot_file(path)
    path = input("Enter the full path of the file to be plotted: ")

