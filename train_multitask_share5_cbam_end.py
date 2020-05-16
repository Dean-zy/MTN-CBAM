# -*- coding: utf-8 -*-
"""
This script implements the training part of the system described in the work

"Keyword Spotting for Hearing Assistive Devices Robust to External Speakers".
Authors: Iván López-Espejo, Zheng-Hua Tan and Jesper Jensen.
INTERSPEECH 2019, September 15-19, Graz (Austria).

In principle, only two variables have to be set by the user below:
    * "DDBB_PATH": This is the directory containing the hearing aid speech
      database. It is expected that there is a folder with noises inside.
    * "DPATH": This is the directory containing the training and validation
      data.

In particular, inside "DPATH" it is expected to find the following files:
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
    * "X_valid.p": This is an Mx101x80 matrix comprising validation input data.
      This consists of 2xM MFCC matrices, with 101 time frames and 40 quefrency
      bins each, from the front and rear mics stacked across the quefrency
      dimension.
    * "Y1_valid.p": As "Y1_train.p" but for the validation dataset.
    * "Y2_valid.p": As "Y2_train.p" but for the validation dataset.

As a result, this script will save a trained multi-task model in "DPATH".
"""


import pickle
import librosa
import random
import numpy as np
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Conv2D, Dense, Activation, BatchNormalization
from keras.layers.merge import add
from keras.activations import relu, softmax, sigmoid
from keras import optimizers
from CBAM import cbam_block,channel_attention,spatial_attention


# ------------------- #
# VARIABLE DEFINITION #
# ------------------- #
DDBB_PATH = './HADataset/' # Google Speech Commands Dataset path.
DPATH = './multitask_dual/' # Working directory.
# ------------------- #


# In order to replicate results a seed is fixed.
np.random.seed(seed=0)
random.seed(0)


# ---------------- #
# MODEL DEFINITION #
# ---------------- #

# Residual block definition.
def res_block(n_fm, layer):
    def f(x):
        dr = int(np.power(2,np.floor(layer/3)))
        h = Conv2D(kernel_size=3, filters=n_fm, dilation_rate=dr, strides=1, use_bias=False, padding='same')(x)
        h = Activation(relu)(h)
        h = BatchNormalization(center=False, scale=False)(h)
        dr = int(np.power(2,np.floor((layer+1)/3)))
        h = Conv2D(kernel_size=3, filters=n_fm, dilation_rate=dr, strides=1, use_bias=False, padding='same')(h)
        h = Activation(relu)(h)
        h = BatchNormalization(center=False, scale=False)(h)
        return add([h, x])
    return f

# Architecture definition.
input_tensor = Input((101, 80, 1))
x = Conv2D(kernel_size=3, filters=45, strides=1, use_bias=False, padding='same')(input_tensor)
x = res_block(45,0)(x)
x = res_block(45,2)(x)
x = res_block(45,4)(x)
x = res_block(45,6)(x)
x = res_block(45,8)(x)
x1 = cbam_block(x)
x1 = res_block(45,10)(x1)
x2= cbam_block(x)
x2 = res_block(45,10)(x2)
x1 = Conv2D(kernel_size=3, filters=45, dilation_rate=16, strides=1, use_bias=False, padding='same')(x1)
x1 = BatchNormalization(center=False, scale=False)(x1)
x1 = GlobalAveragePooling2D()(x1)
x2 = Conv2D(kernel_size=3, filters=45, dilation_rate=16, strides=1, use_bias=False, padding='same')(x2)
x2 = BatchNormalization(center=False, scale=False)(x2)
x2 = GlobalAveragePooling2D()(x2)
x1 = Dense(units=11)(x1)
x1 = Activation(softmax, name='keyword_spotting')(x1)
x2 = Dense(units=1)(x2)
x2 = Activation(sigmoid, name='own_voice_detection')(x2)

# Model compilation.
model = Model(inputs=input_tensor, outputs=[x1, x2])
sgd = optimizers.SGD(lr=0.025, decay=1e-5, momentum=0.9, nesterov=False)
model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'], loss_weights=[1, 1], optimizer=sgd, metrics=['accuracy'])

# ---------------- #


# -------------- #
# MODEL TRAINING #
# -------------- #

# We first prepare the data, which is changed at every epoch.
print("Preparing training data for the first time...")
X1 = pickle.load(open(DPATH + "X_train_front_1.p", "rb"))
X2 = pickle.load(open(DPATH + "X_train_front_2.p", "rb"))
X3 = pickle.load(open(DPATH + "X_train_front_3.p", "rb"))
X_train_WAV_f = np.concatenate((X1, X2, X3), axis=0)
X1 = pickle.load(open(DPATH + "X_train_rear_1.p", "rb"))
X2 = pickle.load(open(DPATH + "X_train_rear_2.p", "rb"))
X3 = pickle.load(open(DPATH + "X_train_rear_3.p", "rb"))
X_train_WAV_r = np.concatenate((X1, X2, X3), axis=0)
Y1_train = pickle.load(open(DPATH + "Y1_train.p", "rb"))
Y2_train = pickle.load(open(DPATH + "Y2_train.p", "rb"))
# Distorting the data for the first time.
no_ts = Y1_train.shape[0] # Number of training examples.
X_train = np.zeros((no_ts, 101, 80)) # Time x frequency.
sr = 16000 # Sampling frequency.
NOISE_PROB = 0.8
DITH = 0.00001 # Dithering scaling factor.
LOGTHR = np.exp(-50)
path_noise = DDBB_PATH + '_background_noise_/' # Background noise path.
noise_list = ['white_noise', 'running_tap', 'pink_noise', 'exercise_bike', 'dude_miaowing', 'doing_the_dishes'] # List of noises.
# Each wavfile is processed.
for i in range(no_ts):
    
    y_f = X_train_WAV_f[i]
    y_r = X_train_WAV_r[i]
    
    # We apply time shifting with Y ~ U[-100,100] ms.
    shift = np.random.uniform(-100,100)
    shift_smpl = int(0.001 * sr * shift) # In terms of number of samples.
    if shift_smpl > 0:
        yf_tmp = np.zeros(sr)
        yr_tmp = np.zeros(sr)
        yf_tmp[:y_f.shape[0]-shift_smpl] = y_f[shift_smpl:]
        yr_tmp[:y_r.shape[0]-shift_smpl] = y_r[shift_smpl:]
        y_f = yf_tmp
        y_r = yr_tmp
    elif shift_smpl < 0:
        shift_smpl = np.absolute(shift_smpl)
        yf_tmp = np.zeros(sr)
        yr_tmp = np.zeros(sr)
        yf_tmp[shift_smpl:] = y_f[:-shift_smpl]
        yr_tmp[shift_smpl:] = y_r[:-shift_smpl]
        y_f = yf_tmp
        y_r = yr_tmp
    
    # We add dithering.
    y_f = y_f + np.random.randn(len(y_f)) * DITH
    y_r = y_r + np.random.randn(len(y_r)) * DITH
        
    # Noise adding.
    if np.random.uniform() < NOISE_PROB:
        noise = random.choice(noise_list)
        n, sr = librosa.load(path_noise + noise + ".wav", sr=None)
        # We cut a noise segment to be summed.
        init_smpl = np.random.randint(0,len(n)-sr)
        ns = n[init_smpl:init_smpl+sr]
        scl_factor = 0.001 * np.random.uniform() # Noise scaling factor.
        y_f = y_f + scl_factor * ns
        y_r = y_r + scl_factor * ns
    
    # To the cepstral domain.
    S_f = librosa.feature.melspectrogram(y=y_f, sr=sr, n_fft=int(0.030*sr), hop_length=int(0.01*sr), n_mels=40, fmin=20, fmax=4000)
    Slog_f = np.log(np.maximum(S_f, LOGTHR))
    mfccs_f = librosa.feature.mfcc(y=None, sr=sr, S=Slog_f, n_mfcc=40)
    S_r = librosa.feature.melspectrogram(y=y_r, sr=sr, n_fft=int(0.030*sr), hop_length=int(0.01*sr), n_mels=40, fmin=20, fmax=4000)
    Slog_r = np.log(np.maximum(S_r, LOGTHR))
    mfccs_r = librosa.feature.mfcc(y=None, sr=sr, S=Slog_r, n_mfcc=40)
    mfccs = np.concatenate((mfccs_f, mfccs_r), axis=0)
    
    # Appending the result.
    X_train[i] = mfccs.T

# Then, we normalize the data and reshape for training.
X_train_final = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_final = X_train_final.reshape(no_ts, 101, 80, 1)
# We load the validation data.
X_valid = pickle.load(open(DPATH + "X_valid.p", "rb"))
X_valid = X_valid.reshape(X_valid.shape[0], 101, 80, 1)
Y1_valid = pickle.load(open(DPATH + "Y1_valid.p", "rb"))
Y2_valid = pickle.load(open(DPATH + "Y2_valid.p", "rb"))
X_valid = (X_valid - np.mean(X_valid)) / np.std(X_valid)

# We train the first iteration.
model.fit(X_train_final, [Y1_train, Y2_train], batch_size=64, validation_data=(X_valid, [Y1_valid, Y2_valid]), epochs=1)

# Then, we iterate epochs in order to train the model and regenerate 30%
# of training data at each epoch.
n_epochs = 26 # We train the model for a total of "n_epochs" epochs.
n_rs = int(0.3*no_ts) # Number of training examples to be regenerated.
for e in range(n_epochs-1):
    
    print("EPOCH: " + str(e+2))
    
    # Training data regeneration.
    print("Regenerating 30% of training data...")
    rv = np.random.permutation(no_ts)
    r_i = rv[:n_rs]
    
    for i in r_i:
        
        y_f = X_train_WAV_f[i]
        y_r = X_train_WAV_r[i]
        
        # We apply time shifting with Y ~ U[-100,100] ms.
        shift = np.random.uniform(-100,100)
        shift_smpl = int(0.001 * sr * shift) # In terms of number of samples.
        if shift_smpl > 0:
            yf_tmp = np.zeros(sr)
            yr_tmp = np.zeros(sr)
            yf_tmp[:y_f.shape[0]-shift_smpl] = y_f[shift_smpl:]
            yr_tmp[:y_r.shape[0]-shift_smpl] = y_r[shift_smpl:]
            y_f = yf_tmp
            y_r = yr_tmp
        elif shift_smpl < 0:
            shift_smpl = np.absolute(shift_smpl)
            yf_tmp = np.zeros(sr)
            yr_tmp = np.zeros(sr)
            yf_tmp[shift_smpl:] = y_f[:-shift_smpl]
            yr_tmp[shift_smpl:] = y_r[:-shift_smpl]
            y_f = yf_tmp
            y_r = yr_tmp
        
        # We add dithering.
        y_f = y_f + np.random.randn(len(y_f)) * DITH
        y_r = y_r + np.random.randn(len(y_r)) * DITH
        
        # Noise adding.
        if np.random.uniform() < NOISE_PROB:
            noise = random.choice(noise_list)
            n, sr = librosa.load(path_noise + noise + ".wav", sr=None)
            # We cut a noise segment to be summed.
            init_smpl = np.random.randint(0,len(n)-sr)
            ns = n[init_smpl:init_smpl+sr]
            scl_factor = 0.001 * np.random.uniform() # Noise scaling factor.
            y_f = y_f + scl_factor * ns
            y_r = y_r + scl_factor * ns
            
        # To the cepstral domain.
        S_f = librosa.feature.melspectrogram(y=y_f, sr=sr, n_fft=int(0.030*sr), hop_length=int(0.01*sr), n_mels=40, fmin=20, fmax=4000)
        Slog_f = np.log(np.maximum(S_f, LOGTHR))
        mfccs_f = librosa.feature.mfcc(y=None, sr=sr, S=Slog_f, n_mfcc=40)
        S_r = librosa.feature.melspectrogram(y=y_r, sr=sr, n_fft=int(0.030*sr), hop_length=int(0.01*sr), n_mels=40, fmin=20, fmax=4000)
        Slog_r = np.log(np.maximum(S_r, LOGTHR))
        mfccs_r = librosa.feature.mfcc(y=None, sr=sr, S=Slog_r, n_mfcc=40)
        mfccs = np.concatenate((mfccs_f, mfccs_r), axis=0)
        
        # Appending the result.
        X_train[i] = mfccs.T
    
    # Then, we normalize the data and reshape for training.
    X_train_final = (X_train - np.mean(X_train)) / np.std(X_train)
    X_train_final = X_train_final.reshape(no_ts, 101, 80, 1)
    # Model training.
    model.fit(X_train_final, [Y1_train, Y2_train], batch_size=64, validation_data=(X_valid, [Y1_valid, Y2_valid]), epochs=1)

# Saving the model.
print("Saving the model...")
model_json = model.to_json()
with open(DPATH + "cbam_end_share5_c3.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(DPATH + "cbam_end_share5_c3.h5")
