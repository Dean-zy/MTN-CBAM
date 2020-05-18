# -*- coding: utf-8 -*-
"""
This script prepares the validation and test data of the system described in
the work

"Keyword Spotting for Hearing Assistive Devices Robust to External Speakers".
Authors: Iván López-Espejo, Zheng-Hua Tan and Jesper Jensen.
INTERSPEECH 2019, September 15-19, Graz (Austria).

In principle, only two variables have to be set by the user below:
    * "DDBB_PATH": This is the directory containing the hearing aid speech
      database.
    * "DPATH": This is the directory where the validation and test data are
      going to be stored.

As a result, this script will save the following files inside "DPATH":
    * "X_valid.p" and "X_test.p": These files contain validation and test
      Nx101x80 dual-channel MFCC matrices, where N is the number of either
      validation or test examples. 101 is the number of time frames and each
      time frame is represented by 40 MFCCs per channel. These data come from
      validation an test waveforms from the front and rear mics of the hearing
      aid. These waveforms correspond to both keywords and non-keywords as well
      as to utterances coming from both the user and external speakers.
    * "Y1_valid.p" and "Y1_test.p": These are Nx11 matrices with one-hot
      encoding vectors, where 10 is the number of keywords (plus the non-
      keyword class, 11) and N is the number of either validation or test
      examples. These matrices are aligned with the "X_valid" and "X_test"
      files mentioned above.
    * "Y2_valid.p" and "Y2_test.p": These are Nx1 matrices containing the
      target data for own-voice/external speaker detection validation and test.
      1 is expected for own voice and 0 for an external speaker. These matrices
      are aligned with the "X_valid" and "X_test" files mentioned above.
    * "angle_valid.p" and "angle_test.p": These are Nx1 matrices containing
      the position (angle) of the external speakers with respect to the user.
"""


import librosa
import numpy as np
import pickle
import random
import glob


# We fix a seed to replicate results.
np.random.seed(seed=0)
random.seed(0)


# ------------------- #
# VARIABLE DEFINITION #
# ------------------- #
DDBB_PATH = './HADataset/' # Hearing aid speech database path.
DPATH = './exp/' # Working directory.
UNK_W = 0.1 # Proportion of the "unknown word" class over the total.
SIL_W = 0 # Proportion of the "silence" class over the total.
LOGTHR = np.exp(-50)
DITH = 0.00001 # Dithering scaling factor.
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


# --------------- #
# VALIDATION DATA #
# --------------- #

with open('Valid_Data.txt') as f:
    files = f.readlines()

# We first determine the number of validation keyword examples to know the
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

no_vs = int(no_kw / (1 - UNK_W - SIL_W)) # Total validation examples.
no_unk = no_vs - no_kw
no_sil = no_vs - no_kw - no_unk # Number of validation examples of 'silence'.
step_unk = int(np.floor(total_unk / no_unk)) # We have to take one 'unknown' sample every step_unk.

# We initialize the variables with the samples.
X_valid = np.zeros((no_vs, 101, 80, 1)) # We expect 101 frames of 40 MFCCs per channel.
Y1_valid = np.zeros((no_vs, NCLASS)) # We will recognize among 11 different classes.
Y2_valid = np.zeros((no_vs, 1)) # This is to detect external speakers.
angle_valid = np.zeros((no_vs, 1)) # This is to store the position (angle) of the external speaker.
ind = 0
ind_t = 0
ind_nkw = 1 # For balance purposes we only take 1 in step_unk unknown keywords.
cont_unk = 0 # To count the number of unknown words already included.

for file in files:
    
    print("Processing validation file " + str(ind_t+1) + "/" + str(len(files)))
    file = file[:-5]
    list_files = glob.glob(DDBB_PATH + file + '*')
    list_files = np.sort(list_files)
    wavfile_f = list_files[0]
    wavfile_r = list_files[2]
    segs = wavfile_f.split("/")
    fname = segs[-1]
    kw = segs[-2]
    tipo = fname.split("_")
    tipo = tipo[3] # To find out if it is an external speaker or not.
    y_f, sr = librosa.load(wavfile_f, sr=None)
    y_r, sr = librosa.load(wavfile_r, sr=None)
    
    # Zero-padding to ensure 1 second long segments.
    if y_f.shape[0] < sr:
        yf_tmp = np.zeros(sr)
        yr_tmp = np.zeros(sr)
        yf_tmp[:y_f.shape[0]] = y_f
        yr_tmp[:y_r.shape[0]] = y_r
        y_f = yf_tmp
        y_r = yr_tmp
    
    # We add dithering.
    y_f = y_f + np.random.randn(len(y_f)) * DITH
    y_r = y_r + np.random.randn(len(y_r)) * DITH
    
    # To the cepstral domain.
    S_f = librosa.feature.melspectrogram(y=y_f, sr=sr, n_fft=int(0.030*sr), hop_length=int(0.01*sr), n_mels=40, fmin=20, fmax=4000)
    Slog_f = np.log(np.maximum(S_f, LOGTHR))
    mfccs_f = librosa.feature.mfcc(y=None, sr=sr, S=Slog_f, n_mfcc=40)
    S_r = librosa.feature.melspectrogram(y=y_r, sr=sr, n_fft=int(0.030*sr), hop_length=int(0.01*sr), n_mels=40, fmin=20, fmax=4000)
    Slog_r = np.log(np.maximum(S_r, LOGTHR))
    mfccs_r = librosa.feature.mfcc(y=None, sr=sr, S=Slog_r, n_mfcc=40)
    mfccs = np.concatenate((mfccs_f,mfccs_r),axis=0)
    
    # We process the output category.
    pos = get_category(kw)
    if pos == 10:
        if (ind_nkw == step_unk) and (cont_unk < no_unk):
            X_valid[ind,:,:,0] = mfccs.T
            Y1_valid[ind,pos] = 1.0
            if tipo[0] == 'O':
                Y2_valid[ind] = 1
            else:
                angle_valid[ind] = int(tipo[2:])
            ind_nkw = 1
            ind += 1
            cont_unk += 1 # A new unknown word sample has been included.
        else:
            ind_nkw += 1
    else:
        X_valid[ind,:,:,0] = mfccs.T
        Y1_valid[ind,pos] = 1.0
        if tipo[0] == 'O':
            Y2_valid[ind] = 1
        else:
            angle_valid[ind] = int(tipo[2:])
        ind += 1
    ind_t += 1

print("Saving validation data...")
pickle.dump(X_valid, open(DPATH + "X_valid.p", "wb"))
pickle.dump(Y1_valid, open(DPATH + "Y1_valid.p", "wb"))
pickle.dump(Y2_valid, open(DPATH + "Y2_valid.p", "wb"))
pickle.dump(angle_valid, open(DPATH + "angle_valid.p", "wb"))


# --------- #
# TEST DATA #
# --------- #

with open('Test_Data.txt') as f:
    files = f.readlines()

# We first determine the number of test keyword examples to know the
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

no_ts = int(no_kw / (1 - UNK_W - SIL_W)) # Total test examples.
no_unk = no_ts - no_kw
no_sil = no_ts - no_kw - no_unk # Number of test examples of 'silence'.
step_unk = int(np.floor(total_unk / no_unk)) # We have to take one 'unknown' sample every step_unk.

# We initialize the variables with the samples.
X_test = np.zeros((no_ts, 101, 80, 1)) # We expect 101 frames of 40 MFCCs per channel.
Y1_test = np.zeros((no_ts, NCLASS)) # We will recognize among 11 different classes.
Y2_test = np.zeros((no_ts, 1)) # This is to detect interferences.
angle_test = np.zeros((no_ts, 1)) # This is to store the position (angle) of the external speaker.
ind = 0
ind_t = 0
ind_nkw = 1 # For balance purposes we only take 1 in step_unk unknown keywords.
cont_unk = 0 # To count the number of unknown words already included.

for file in files:
    
    print("Processing test " + str(ind_t+1) + "/" + str(len(files)))
    file = file[:-5]
    list_files = glob.glob(DDBB_PATH + file + '*')
    list_files = np.sort(list_files)
    wavfile_f = list_files[0]
    wavfile_r = list_files[2]
    segs = wavfile_f.split("/")
    fname = segs[-1]
    kw = segs[-2]
    tipo = fname.split("_")
    tipo = tipo[3] # To find out if it is an external speaker or not.
    y_f, sr = librosa.load(wavfile_f, sr=None)
    y_r, sr = librosa.load(wavfile_r, sr=None)
    
    # Zero-padding to ensure 1 second long segments.
    if y_f.shape[0] < sr:
        yf_tmp = np.zeros(sr)
        yr_tmp = np.zeros(sr)
        yf_tmp[:y_f.shape[0]] = y_f
        yr_tmp[:y_r.shape[0]] = y_r
        y_f = yf_tmp
        y_r = yr_tmp
    
    # We add dithering.
    y_f = y_f + np.random.randn(len(y_f)) * DITH
    y_r = y_r + np.random.randn(len(y_r)) * DITH
    
    # To the cepstral domain.
    S_f = librosa.feature.melspectrogram(y=y_f, sr=sr, n_fft=int(0.030*sr), hop_length=int(0.01*sr), n_mels=40, fmin=20, fmax=4000)
    Slog_f = np.log(np.maximum(S_f, LOGTHR))
    mfccs_f = librosa.feature.mfcc(y=None, sr=sr, S=Slog_f, n_mfcc=40)
    S_r = librosa.feature.melspectrogram(y=y_r, sr=sr, n_fft=int(0.030*sr), hop_length=int(0.01*sr), n_mels=40, fmin=20, fmax=4000)
    Slog_r = np.log(np.maximum(S_r, LOGTHR))
    mfccs_r = librosa.feature.mfcc(y=None, sr=sr, S=Slog_r, n_mfcc=40)
    mfccs = np.concatenate((mfccs_f,mfccs_r),axis=0)
    
    # We process the output category.
    pos = get_category(kw)
    if pos == 10:
        if (ind_nkw == step_unk) and (cont_unk < no_unk):
            X_test[ind,:,:,0] = mfccs.T
            Y1_test[ind,pos] = 1.0
            if tipo[0] == 'O':
                Y2_test[ind] = 1
            else:
                angle_test[ind] = int(tipo[2:])
            ind_nkw = 1
            ind += 1
            cont_unk += 1 # A new unknown word sample has been included.
        else:
            ind_nkw += 1
    else:
        X_test[ind,:,:,0] = mfccs.T
        Y1_test[ind,pos] = 1.0
        if tipo[0] == 'O':
            Y2_test[ind] = 1
        else:
            angle_test[ind] = int(tipo[2:])
        ind += 1
    ind_t += 1

print("Saving test data...")
pickle.dump(X_test, open(DPATH + "X_test.p", "wb"))
pickle.dump(Y1_test, open(DPATH + "Y1_test.p", "wb"))
pickle.dump(Y2_test, open(DPATH + "Y2_test.p", "wb"))
pickle.dump(angle_test, open(DPATH + "angle_test.p", "wb"))