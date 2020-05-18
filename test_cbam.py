#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script implements the testing part of the system described in the work

"Keyword Spotting for Hearing Assistive Devices Robust to External Speakers".
Authors: Iván López-Espejo, Zheng-Hua Tan and Jesper Jensen.
INTERSPEECH 2019, September 15-19, Graz (Austria).

In principle, only one variable has to be set by the user below:
    * "DPATH": This is the directory containing the testing data along with
      the trained multi-task model.

In particular, inside "DPATH" it is expected to find the following files:
    * "multitask_dual.json" and "multitask_dual.h5": This is the trained multi-
      task model. To obtain it, run the script "train_multitask_dual.py".
    * "X_test.p": This is an Mx101x80 matrix comprising test input data. This
      consists of 2xM MFCC matrices, with 101 time frames and 40 quefrency bins
      each, from the front and rear mics stacked across the quefrency
      dimension.
    * "Y1_test.p": This is an Mx11 matrix with one-hot encoding vectors, where
      10 is the number of keywords (plus the non-keyword class, 11) and M is
      the number of test examples. This matrix is aligned with "X_test.p".
    * "Y2_test.p": This is an Mx1 matrix containing the target data for
      own-voice/external speaker detection testing. 1 is expected for own
      voice and 0 for an external speaker. This matrix is aligned with
      "X_test.p".
    * "angle_test.p": This is an Mx1 matrix containing the index of the
      external speaker angle with respect to the hearing aid user. Index is 0
      at 0°, 1 at 7.5°, ..., and 47 at 352.5° (counterclockwise). Own-voice
      samples have been assigned a 0 value as well (just for filling).

As a result, this script will display some accuracy results on the screen.
"""


import pickle
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# ------------------- #
# VARIABLE DEFINITION #
# ------------------- #
DPATH = './exp/'
# ------------------- #


# ---------------------- #
# MODEL AND DATA LOADING #
# ---------------------- #
modelname=sys.argv[1]
json_file = open(DPATH + modelname + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Loading weights into a new model.
loaded_model.load_weights(DPATH + modelname + ".h5")
print("Loaded model from disk...")

# We then load the test data.
X_test = pickle.load(open(DPATH + "/X_test.p", "rb"))
Y1_test = pickle.load(open(DPATH + "/Y1_test.p", "rb"))
Y2_test = pickle.load(open(DPATH + "/Y2_test.p", "rb"))
angle_test = pickle.load(open(DPATH + "/angle_test.p", "rb"))
# Reshaping and normalization.
X_test = X_test.reshape(X_test.shape[0], 101, 80, 1)
X_test = (X_test - np.mean(X_test)) / np.std(X_test)

# ---------------------- #


# ---------- #
# PREDICTION #
# ---------- #

predicted = loaded_model.predict(X_test)
spotting = np.asarray(predicted[0])
speaker = np.asarray(predicted[1])
speaker = speaker > 0.5

# Global accuracy computation.
err = 0
acc_spk = 0
amax_v = spotting.argmax(axis=1)
err_v = amax_v - Y1_test.argmax(axis=1)
acc_temp = Y2_test - speaker
TOTAL = len(err_v)
for i in range(TOTAL):
    if Y2_test[i] == 1:
        if err_v[i] != 0 or speaker[i] == 0:
            err += 1
    else:
        if amax_v[i] < 10 and speaker[i] == 1:
            err += 1
    if acc_temp[i] == 0:
        acc_spk += 1

err = err / TOTAL
acc_spk = acc_spk / TOTAL
print("Overall KWS Accuracy: " + str(100*(1-err)) + "%")
print("Overall Own-voice/External Speaker Detection Accuracy: " + str(100*acc_spk) + "%")

# We separate the variables for an easier procedure.
no_ov = int(np.sum(Y2_test)) # No. of own-voice examples.
no_ex = len(Y2_test) - no_ov # No. of external examples.
Y1_test_ov = np.zeros((no_ov,11))
angle_ex = np.zeros((no_ex,1))
spotting_ov = np.zeros((no_ov,11))
speaker_ov = np.zeros((no_ov,1))
speaker_ex = np.zeros((no_ex,1))
i_ov = 0
i_ex = 0
for i in range(TOTAL):
    if Y2_test[i] == 1:
        # Own-voice.
        Y1_test_ov[i_ov,:] = Y1_test[i,:]
        spotting_ov[i_ov,:] = spotting[i,:]
        speaker_ov[i_ov] = speaker[i]
        i_ov += 1
    else:
        # External speaker.
        angle_ex[i_ex] = angle_test[i]
        speaker_ex[i_ex] = speaker[i]
        i_ex += 1

# Accuracy computation.
acc_ov = 0
acc_v_ov = spotting_ov.argmax(axis=1) - Y1_test_ov.argmax(axis=1)
TOTAL_ov = len(acc_v_ov)
for i in range(TOTAL_ov):
    if acc_v_ov[i] == 0:
        acc_ov += 1

acc_ov = acc_ov / TOTAL_ov
intov = np.sum(speaker_ov)/len(speaker_ov)
intex = np.sum(1-speaker_ex)/len(speaker_ex)
print("Own-voice KWS Accuracy: " + str(100*acc_ov) + "%")
print("Own-voice Detection Accuracy: " + str(100*intov) + "%")
print("External Speaker Detection Accuracy: " + str(100*intex) + "%")

# ---------- #


# --------------------- #
# ANGLE PLOT GENERATION #
# --------------------- #

# Finally, we study external speaker detection as a function of the angle.
correct = np.zeros((48,1)) # We have 48 different angles.
total = np.zeros((48,1))
for i in range(len(speaker_ex)):
    # We retrieve the angle.
    ang = int(angle_ex[i])
    total[ang] += 1
    if speaker_ex[i] == 0:
        correct[ang] += 1

step = 7.5*np.pi/180
theta = np.arange(0,2*np.pi+step,step)
r = correct / total
r = np.append(r,r[0])

# We finally compute a smoothed version.
w = np.array([0.1, 0.2, 0.4, 0.2, 0.1]) # Weights.
r_s = np.zeros((len(r)-1,1))
r_s[0] = w[0]*r[-3] + w[1]*r[-2] + np.sum(w[2:]*r[0:3])
r_s[1] = w[0]*r[-2] + np.sum(w[1:]*r[0:4])
for i in range(len(r_s)-3):
    r_s[i+2] = np.sum(w*r[i:i+5])
r_s[-1] = np.sum(w[:-1]*r[-4:]) + w[-1]*r[1]
r_s = np.append(r_s,r_s[0])

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8],polar=True)
ax.set_ylim(0,1)
ax.set_yticks(np.arange(0.2,1,0.2))
ax.plot(theta,r_s,lw=2.5)
plt.legend(['External spk. detection accuracy'])
# plt.show()