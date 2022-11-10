import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

import w2v2_predict
from utils.vad import *

from UnsupSeg38 import predict as seg_pred

import sys

import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn import metrics

import soundfile as sf

## Based off code: https://dsp.stackexchange.com/questions/72027/python-audio-analysis-which-spectrogram-should-i-use-and-why
def wavToSpec(wav_path):
    samplingFrequency, signalData = wavfile.read(wav_path)
    # Plot the signal read from wav file
    plt.subplot(111)
    plt.specgram(signalData, Fs=samplingFrequency)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

def display_segs(wav_path, segVect):
    samplingFrequency, signalData = wavfile.read(wav_path)
    # Plot the signal read from wav file
    plt.subplot(111)
    plt.specgram(signalData, Fs=samplingFrequency)
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    for seg in segVect:
        plt.vlines(seg, ymin=0, ymax=samplingFrequency/2, colors = "yellow")

    plt.show()

def displayFA(segList, wav_path):
    samplingFrequency, signalData = wavfile.read(wav_path)
    # Plot the signal read from wav file
    plt.figure(figsize=(15, 4))
    plt.subplot(111)
    plt.specgram(signalData, Fs=samplingFrequency)
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    for seg in segList:
        plt.vlines(seg["start"], ymin=0, ymax=samplingFrequency / 2, colors="yellow", linestyles="dashed")

    for seg in segList:
        plt.text((seg["stop"]+seg["start"])/2 - 0.015, samplingFrequency / 4 , seg["phone"], fontsize=12)

    plt.show()

def FA_demo(wav_path, vad = True, clean = True, clean_agg = True, bias = 0.5, display = True):
    # Get the tokens
    demo_pickle_path = './demo.pickle'
    if os.path.isfile(demo_pickle_path):
        infile = open(demo_pickle_path, 'rb')
        wp = pickle.load(infile)
        infile.close()
    else:
        wp = w2v2_predict.w2v2_predictor()
        wp.set_model(ckpt="./wav2vec2-base-timit-demo-phones/checkpoint-11500")

        outfile = open(demo_pickle_path, 'wb')
        pickle.dump(wp, outfile)
        outfile.close()

    signalData, samplingFrequency = sf.read(wav_path)

    # Length of speech in seconds
    seconds = len(signalData) / samplingFrequency

    # Get the precollapse tokens
    tokens = wp.pred_wav_no_collapse(wav_path, return_type="phones")

    # Use Unsupseg38 by felix kreuk to get the segmentations
    segVect = seg_pred.pred(wav=wav_path,
                            ckpt='./UnsupSeg38/runs/2021-09-26_16-45-28-default/epoch=4.ckpt',
                            prominence=None)

    #   Of an array of 0's and 1's, return the time in seconds when the transitions occur
    #   arr: an array of zeroes and ones
    #   st: sample length in time (s)
    def detectEdges(arr, st):
        tmp = None
        tmp2 = None
        return_array = list()

        for ii in range(len(arr)):
            jj = int(ii)
            if tmp == None:
                # Condtion: we just instantiated the array
                tmp = arr[jj][0]

            # Condition: rising edge
            elif arr[jj][0] != tmp and tmp == 0:
                # Store timing of rising edge in seconds
                tmp2 = jj * st / len(arr)

                tmp = 1

            # Condition: falling edge
            elif arr[jj][0] != tmp and tmp == 1:
                # add the pair of boundaries to the return list
                return_array.append((tmp2, jj * st / len(arr)))
                tmp = 0

        return return_array

    #   removes segmentations which are deemed to have
    #   occured in spaces of audio where there is no speech
    #
    #   Signal:     signal data to process
    #   sr:         sampling rate
    #   tolerance: tolerance in difference in VAD boundaries and Seg boundaries difference
    def filterSegmentations(segmentations, signal, sr, tolerance=0.05):
        vad = VAD(signal, sr, nFFT=2048, win_length=0.025, hop_length=0.01, theshold=0.5)
        vad = vad.astype('int')
        vad = vad.tolist()
        vadEdges = detectEdges(vad, len(signal) / sr)

        filtered_segs = list()
        # Scan through all the segmentations looking for segments which fit withing the boundaries.
        # Works in O(N^2) time because im a pig.
        # filtered_segs.append(0)
        for seg in segmentations:
            for vadBound in vadEdges:
                if vadBound[0] - tolerance <= seg and seg <= vadBound[1] + tolerance:
                    filtered_segs.append(seg)
        # filtered_segs.append(len(signal)/sr)

        return filtered_segs

    def filterSegmentationsWrapper(wav_path, segmentations):
        signal, sr = sf.read(wav_path)
        return filterSegmentations(segmentations, signal, sr)



    if vad == True:
        segVect = filterSegmentationsWrapper(wav_path=wav_path,
                                             segmentations=segVect)
    else:
        segVect = segVect.tolist()

    def tokens_to_timedtokens(signalData, samplingFrequency, tokens):
        # Duration of utterance in seconds
        seconds = len(signalData) / samplingFrequency

        # Delta s is half the distance in time between each token
        delta_s = seconds / (2 * len(tokens))

        # A list of tokens with time attached. It's called votelist because itll do some voting later on
        timed_token_list = list()

        # instantiate timestamp with one delta s. The distance between each token in time is 2 times delta_s
        timestamp = delta_s

        # This for loop creats a list of tuples with the timing attached to each
        for token in tokens:
            # Timed token is a tuple with the time in the sequence at which it occurs
            timed_token = (token, timestamp)

            # Add timed token to the voter list
            timed_token_list.append(timed_token)

            # Increment the timestamp for the next token
            timestamp = timestamp + 2 * delta_s
        return timed_token_list

    timed_token_list = tokens_to_timedtokens(signalData, samplingFrequency, tokens)

    # Now keep only the labels worth interpreting
    filtered_time_token_list = list()
    for tt in timed_token_list:
        if tt[0] != ("[PAD]" or "[UNK]" or "|"):
            filtered_time_token_list.append(tt)


    # Compute Decision Boundaries
    def decision_boundary_calc(filtered_time_token_list, seconds, bias=0.5):
        assert 0 <= bias and bias <= 1
        DCB = list()
        for ii in range(len(filtered_time_token_list)):
            if ii == len(filtered_time_token_list) - 1:  # CASE: Last token
                upper = seconds
                lower = (filtered_time_token_list[ii - 1][1]) * (1 - bias) + (filtered_time_token_list[ii][1]) * (bias)
            elif ii == 0:  # CASE: First token
                upper = filtered_time_token_list[ii + 1][1] * (bias) + filtered_time_token_list[ii][1] * (1 - bias)
                lower = 0
            else:
                upper = (filtered_time_token_list[ii + 1][1]) * (bias) + (filtered_time_token_list[ii][1]) * (1 - bias)
                lower = (filtered_time_token_list[ii - 1][1]) * (1 - bias) + (filtered_time_token_list[ii][1]) * (bias)
            # append phone label, start time, end time tuple
            DCB.append((filtered_time_token_list[ii][0], lower, upper))
        return DCB

    DCB = decision_boundary_calc(filtered_time_token_list, seconds, bias)

    import json
    try:
        with open('str_unic.json') as str_unic_file:
            str_to_unicode_dict = json.loads(str_unic_file.read())
        Max_DCB_init_dict = dict.fromkeys(str_to_unicode_dict, 0)
    except Exception:
        print("Failed to open str_unic.json")

    Max_DCB_init_dict = dict.fromkeys(str_to_unicode_dict, 0)

    # There are comments in the VAD function to use add this there. But you shouldnt.
    segVect.insert(0, 0)
    segVect.append(seconds)

    def maxContribution(segVect, Max_DCB_init_dict, DCB):
        label_list = list()
        # Find the maximum contributer for each segment
        for segIndex in range(len(segVect)):
            label_dict = Max_DCB_init_dict.copy()

            if segIndex != (len(segVect) - 1):
                t_segStart = segVect[segIndex]
                t_segEnd = segVect[segIndex + 1]
                for dcb in DCB:
                    # CASE: decision starts within the segment
                    if t_segStart <= dcb[1] and dcb[1] <= t_segEnd:
                        # CASE: Decision contained entirely within the segment
                        if dcb[2] <= t_segEnd:
                            label_dict[dcb[0]] = label_dict[dcb[0]] + (dcb[2] - dcb[1])
                        else:  # CASE: Decision starts within the segment but ends elsewhere
                            label_dict[dcb[0]] = label_dict[dcb[0]] + (t_segEnd - dcb[1])
                    # CASE: Decision ends within the segment, but does not start within the segmnet
                    elif t_segStart <= dcb[2] and dcb[2] <= t_segEnd:
                        label_dict[dcb[0]] = label_dict[dcb[0]] + (dcb[2] - t_segStart)
                    # CASE: Decision contains the entirety of the seg
                    elif dcb[1] <= t_segStart and t_segEnd <= dcb[2]:
                        label_dict[dcb[0]] = label_dict[dcb[0]] + (t_segEnd - t_segStart)
                    else:
                        pass
                    # dcb[0] : phone label. dcb[1] : start time, dcb[2] : end time

                label_list.append(max(label_dict, key=label_dict.get))
        return label_list

    label_list = maxContribution(segVect, Max_DCB_init_dict, DCB)

    # Lets zip each label with its start and end times
    segList = list()
    [segList.append((label_list[ii], segVect[ii], segVect[ii + 1])) for ii in range(len(label_list))]

    def clean_segs(segList_in, wav_path):
        segList = segList_in.copy()
        tokens_collapsed = wp.pred_wav_with_collapse(wav_path)
        print("W2V2-CTC Output:", tokens_collapsed)

        transitions = list()
        for ii in range(len(tokens_collapsed) - 1):
            transitions.append((tokens_collapsed[ii], tokens_collapsed[ii + 1]))

        index = 0
        for jj in range(len(transitions)):
            found = False
            limitreached = False

            while found == False and limitreached == False:
                if index >= len(segList) - 1:
                    limitreached = True
                else:
                    seg_from = segList[index]
                    seg_to = segList[index + 1]

                    # CASE: the two elements are the same, ie seglist: aab, transition: ab, focal:aa, turn seglist into ab
                    if seg_from[0] == seg_to[0] and seg_from[0] == transitions[jj][0]:
                        segList[index] = (segList[index][0], segList[index][1], segList[index + 1][2])
                        segList.remove(segList[index + 1])


                    # CASE: Transition is found
                    # This should probably be transitions[jj+1][0], not transitions[jj][1]. Too late to change it.
                    elif seg_from[0] == transitions[jj][0] and seg_to[0] == transitions[jj][1]:
                        found = True
                        index = index + 1

                    else:
                        index = index + 1
                        break

        return segList

    def clean_segs_aggressive(segList_in, wav_path):
        segList = segList_in.copy()
        tokens_collapsed = wp.pred_wav_with_collapse(wav_path)
        print("W2V2-CTC Output:", tokens_collapsed)

        transitions = list()
        for ii in range(len(tokens_collapsed) - 1):
            transitions.append((tokens_collapsed[ii], tokens_collapsed[ii + 1]))

        ceiling = len(segList) - 2
        jj = 0
        finished = False
        while finished == False:
            if jj <= ceiling:
                if segList[jj][0] == segList[jj + 1][0]:
                    if not (segList[jj][0], segList[jj + 1][0]) in transitions:
                        newSeg = (segList[jj][0], segList[jj][1], segList[jj + 1][2])
                        segList[jj] = newSeg
                        segList.remove(segList[jj + 1])
                        ceiling = ceiling - 1
                    jj = jj -1
                jj = jj + 1
            else:
                finished = True

        return segList

    if clean == True:
        if clean_agg == True:
            segList = clean_segs_aggressive(segList, wav_path)
        else:
            segList = clean_segs(segList, wav_path)

    temp = list()
    for seg in segList:
        temp.append({"phone": seg[0],
                     "start": seg[1],
                     "stop": seg[2]})
    segList = temp

    if display == True:
        displayFA(segList, wav_path)
    return segList