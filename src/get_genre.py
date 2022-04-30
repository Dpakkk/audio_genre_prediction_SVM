'''
This is the main file which we used to check the actual genre of mp3 file.
'''


# library here
import pandas as pd
import numpy as np
import torch
import sys
import sklearn
#import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from librosa.core import load
from librosa.feature import melspectrogram
from librosa import power_to_db

# calling our build-up model
from model import genreNet
from config import MODELPATH
from config import GENRES

import warnings
warnings.filterwarnings("ignore")


def separate_by_class(audio_path):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
    return separated


def main(argv):

    if len(argv) != 1:
        print("run this file with the audio path")
        exit()

    le = LabelEncoder().fit(GENRES)
    ## load trained model
    net = genreNet()
    net.load_state_dict(torch.load(MODELPATH, map_location='cpu'))
    ## load audio
    audio_path = argv[0]
    y, sr = load(audio_path, mono=True, sr=22050)
    ## chunks of audio spec
    S = melspectrogram(y, sr).T
    S  = S[:-1 * (S.shape[0] % 128)]
    num_chunk = S.shape[0] / 128
    data_chunks = np.split(S, num_chunk)
    ## classify spec
    genres = list()
    
    # Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated
 
# Test separating data by class
separated = separate_by_class(dataset)
for label in separated:
	print(label)
	for row in separated[label]:
		print(row)
    
    for i, data in enumerate(data_chunks):
        data = torch.FloatTensor(data).view(1, 1, 128, 128)
        preds = net(data)
        pred_val, pred_index = preds.max(1)
        pred_index = pred_index.data.numpy()
        pred_val = np.exp(pred_val.data.numpy()[0])
        pred_genre = le.inverse_transform(pred_index).item()
        if pred_val >= 0.5:
            genres.append(pred_genre)
    s = float(sum([v for k,v in dict(Counter(genres)).items()]))
    pos_genre = sorted([(k, v/s*100 ) for k,v in dict(Counter(genres)).items()], key=lambda x:x[1], reverse=True)
    for genre, pos in pos_genre:
        print("%10s: \t%.2f\t%%" % (genre, pos))
    return

if __name__ == '__main__':
    main(sys.argv[1:])
