# -*- coding: utf-8 -*-
"""
This script prepares the training data for the system described in the work

"Keyword Spotting for Hearing Assistive Devices Robust to External Speakers".
Authors: Iván López-Espejo, Zheng-Hua Tan and Jesper Jensen.
INTERSPEECH 2019, September 15-19, Graz (Austria).

In principle, only two variables have to be set by the user below:
    * "DDBB_PATH": This is the directory containing the hearing aid speech
      database.
    * "DPATH": This is the working directory where the training data are going
      to be stored.

As a result, this script will save the following files inside "DPATH":
    * "X_train_front_1.p", "X_train_front_2.p" and "X_train_front_3.p": These
      files contain training waveforms from the front mic of the hearing aid.
      These waveforms correspond to both keywords and non-keywords as well as
      to utterances coming from both the user and external speakers.
    * "X_train_rear_1.p", "X_train_rear_2.p" and "X_train_rear_3.p": As the
      files above but the waveforms come from the rear mic of the hearing aid.
    * "Y1_train.p": This is an Nx11 matrix with one-hot encoding vectors, where
      10 is the number of keywords (plus the non-keyword class, 11) and N is
      the number of training examples. This matrix is aligned with the
      "X_train" files mentioned above.
    * "Y2_train.p": This is an Nx1 matrix containing the target data for
      own-voice/external speaker detection training. 1 is expected for own
      voice and 0 for an external speaker. This matrix is aligned with the
      "X_train" files mentioned above.
"""


import librosa
import numpy as np
import pickle
import glob


# ------------------- #
# VARIABLE DEFINITION #
# ------------------- #
DDBB_PATH = './HADataset/' # Hearing aid speech database path.
DPATH = './exp/' # Working directory.
UNK_W = 0.1 # Proportion of the "unknown word" class over the total.
SIL_W = 0 # Proportion of the "silence" class over the total.
SRATE = 16000 # Sampling rate of 16 kHz.
NCLASS = 11 # No. of classes.
# ------------------- #


# We discriminate among 11 classes: yes, no, up, down, left, right, on, off,
# stop, go and 'unknown'.
def get_category(keyword):
    if keyword == 'yes':
        cat = 0
    elif keyword == 'no':
        cat = 1
    elif keyword == 'up':
        cat = 2
    elif keyword == 'down':
        cat = 3
    elif keyword == 'left':
        cat = 4
    elif keyword == 'right':
        cat = 5
    elif keyword == 'on':
        cat = 6
    elif keyword == 'off':
        cat = 7
    elif keyword == 'stop':
        cat = 8
    elif keyword == 'go':
        cat = 9
    else:
        cat = 10
    return cat


# -------------------- #
# MAKING TRAINING DATA #
# -------------------- #

with open('Training_Data.txt') as f:
    files = f.readlines()

# We first determine the number of training keyword examples to know the
# number of samples from the 'unknown' and 'silence' classes.
no_kw = 0
total_unk = 0
for file in files:
    kw, fl = file.split("/")
    cat = get_category(kw)
    if cat < 10:
        no_kw += 1
    else:
        total_unk += 1

no_ts = int(no_kw / (1 - UNK_W - SIL_W)) # Total training examples.
no_unk = no_ts - no_kw # To ensure we have no silence.
no_sil = no_ts - no_kw - no_unk # No. of training examples of 'silence'.
step_unk = int(np.floor(total_unk / no_unk)) # We have to take one 'unknown' sample every step_unk.

# We process the front and rear channels.
for chan in ['front', 'rear']:
    
    print("Processing the " + chan + "channel data...")
    
    # We initialize the variables with the samples.
    X_train_WAV = np.zeros((no_ts, SRATE)) # We expect 1 second long utterances.
    Y1_train = np.zeros((no_ts, NCLASS)) # We will recognize among 11 different classes.
    Y2_train = np.zeros((no_ts, 1)) # This is to detect external speakers.
    ind = 0
    ind_t = 0
    ind_nkw = 1 # For balance purposes we only take 1 in step_unk unknown words.
    cont_unk = 0 # To count the number of unknown words already included.
    # If front, no_chan = 0, if rear, no_chan = 2.
    no_chan = 0
    if chan == 'rear':
        no_chan = 2
    
    # File by file processing...
    for file in files:
        
        print("Processing training file " + str(ind_t+1) + "/" + str(len(files)))
        file = file[:-5]
        list_files = glob.glob(DDBB_PATH + file + '*')
        list_files = np.sort(list_files)
        wavfile = list_files[no_chan]
        segs = wavfile.split("/")
        fname = segs[-1]
        kw = segs[-2]
        tipo = fname.split("_")
        tipo = tipo[3] # To find out if it is an external speaker or not.
        y, sr = librosa.load(wavfile, sr=None)
        
        # Zero-padding to ensure 1 second long segments.
        if y.shape[0] < sr:
            y_tmp = np.zeros(sr)
            y_tmp[:y.shape[0]] = y
            y = y_tmp
        
        # We process the output category.
        pos = get_category(kw)
        if pos == 10:
            if (ind_nkw == step_unk) and (cont_unk < no_unk):
                X_train_WAV[ind] = y
                Y1_train[ind,pos] = 1.0
                if tipo[0] == 'O':
                    Y2_train[ind] = 1
                ind_nkw = 1
                ind += 1
                cont_unk += 1 # A new unknown word sample has been included.
            else:
                ind_nkw += 1
        else:
            X_train_WAV[ind] = y
            Y1_train[ind,pos] = 1.0
            if tipo[0] == 'O':
                Y2_train[ind] = 1
            ind += 1
        ind_t += 1
    
    # We split the resulting matrix as is huge.
    step = int(no_ts/3)
    X_train_WAV_1 = X_train_WAV[:step]
    X_train_WAV_2 = X_train_WAV[step:2*step]
    X_train_WAV_3 = X_train_WAV[2*step:]
    
    print("Saving WAV training data...")
    pickle.dump(X_train_WAV_1, open(DPATH + "X_train_" + chan + "_1.p", "wb"))
    pickle.dump(X_train_WAV_2, open(DPATH + "X_train_" + chan + "_2.p", "wb"))
    pickle.dump(X_train_WAV_3, open(DPATH + "X_train_" + chan + "_3.p", "wb"))
    pickle.dump(Y1_train, open(DPATH + "Y1_train.p", "wb"))
    pickle.dump(Y2_train, open(DPATH + "Y2_train.p", "wb"))