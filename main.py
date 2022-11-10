import os.path

import utilities

import w2v2_predict
from utils.vad import *

from UnsupSeg38 import predict as seg_pred

from datasets import load_dataset, load_metric

from cloudstor import cloudstor

import shutil

import sys

import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn import metrics

import soundfile as sf


def main():
    #######################
    # PARAM DECLARATION   #
    #######################
    bias = 0.5
    use_vad = True
    use_clean = True
    #False for midpoints, True for onset boundaries
    AIE_evaluation = False
    clean_aggressive = True
    experiment_name = "HEY"
    experiment_path = './Experiments/'+experiment_name


    if not os.path.isdir(experiment_path+'/'):
        os.mkdir(experiment_path)

    #Lets evaluate the ensembler
    timit = load_dataset("timit_asr")

    timit = timit.remove_columns(["word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

    import json

    with open('vocab.json') as vocab_file:
        unicode_to_numeric_dict = json.loads(vocab_file.read())
    with open('str_unic.json') as str_unic_file:
        str_to_unicode_dict = json.loads(str_unic_file.read())

    def to_unicode_fn(batch):
        aux_lst = []
        for detailed_utterance in batch['phonetic_detail']:
            lst = []
            for phone in detailed_utterance['utterance']:
                lst.append(str_to_unicode_dict[phone])
            detailed_utterance['unic_utterance'] = lst[:]
            aux_lst.append(detailed_utterance)
        batch['phonetic_detail'] = aux_lst[:]
        return batch

    timit = timit.map(to_unicode_fn, batch_size=-1, batched=True)

    ## CONVERT LIST OF PHONES TO STRING OF PHONES
    def delim_phones_fn(batch):
        for detailed_utterance in batch['phonetic_detail']:
            #detailed_utterance['string_utterance'] = '|'.join(detailed_utterance['unic_utterance'])
            detailed_utterance['string_utterance'] = ''.join(detailed_utterance['unic_utterance'])
        return batch
    timit = timit.map(delim_phones_fn, batch_size=-1, batched = True)


    ## CONVERSION TO 1D ARR
    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = sf.read(batch["file"])
        batch["speech"] = speech_array
        batch["sampling_rate"] = sampling_rate

        targ_seg_list = list()

        for ii in range(len(batch['phonetic_detail']['start'])):
            targ_seg_list.append({"phone":batch['phonetic_detail']['utterance'][ii],
                                  "start":batch['phonetic_detail']['start'][ii]/sampling_rate,
                                  "stop":batch['phonetic_detail']['stop'][ii]/sampling_rate,
                                  "midpoint":(batch['phonetic_detail']['start'][ii]+batch['phonetic_detail']['stop'][ii])/(2*sampling_rate)
                                  })


        batch["target_segs"] = targ_seg_list
        return batch

    timit = timit.map(speech_file_to_array_fn, keep_in_memory=True, num_proc=8)
    timit

    ## Takes :
    ## wav file
    def seg_demo(wav_path):
        signalData, samplingFrequency = sf.read(wav_path)

        # Duration of utterance in seconds
        seconds = len(signalData) / samplingFrequency

        # Get the tokens
        wp = w2v2_predict.w2v2_predictor()
        wp.set_model(
            #ckpt="/home/bryce/PycharmProjects/EnsemblePhoneme/wav2vec2-base-timit-demo-phones/checkpoint-11500")
            ckpt="./wav2vec2-base-timit-demo-phones/checkpoint-11500")
        # tokens = ['h#', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'hh', '[PAD]', '[PAD]', '[PAD]', 'eh', '[PAD]', '[PAD]', 'l', '[PAD]', '[PAD]', 'ow', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'pau', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'dh', 'dh', '[PAD]', '[PAD]', 'ih', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 's', '[PAD]', '[PAD]', 'q', '[PAD]', '[PAD]', 'ih', '[PAD]', '[PAD]', '[PAD]', 'z', '[PAD]', '[PAD]', 'ix', '[PAD]', '[PAD]', 'tcl', 'tcl', 't', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'eh', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 's', '[PAD]', '[PAD]', 'tcl', '[PAD]', 't', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'h#']
        tokens = wp.pred_wav_no_collapse(wav_path, return_type="phones")

        # Use Unsupseg38 by felix kreuk to get the segmentations
        segVect = filterSegmentationsWrapper(wav_path=wav_path,
                                             segmentations=seg_pred.pred(wav=wav_path,
                                                                         ckpt='./UnsupSeg38/runs/2021-09-26_16-45-28-default/epoch=4.ckpt',
                                                                         prominence=None))

        # Delta s is half the distance in time between each token
        delta_s = seconds / (2 * len(tokens))

        # A list of tokens with time attached.
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

        # Now keep only the labels worth interpreting
        filtered_time_token_list = list()
        for tt in timed_token_list:
            if tt[0] != ("[PAD]" or "[UNK]" or "|"):
                filtered_time_token_list.append(tt)

        # Compute Decision Boundaries
        def decision_boundary_calc(filtered_time_token_list, seconds):
            DCB = list()
            for ii in range(len(filtered_time_token_list)):
                if ii == len(filtered_time_token_list) - 1:  # CASE: Last token
                    upper = seconds
                    lower = (filtered_time_token_list[ii - 1][1] + filtered_time_token_list[ii][1]) / 2
                elif ii == 0:  # CASE: First token
                    upper = (filtered_time_token_list[ii + 1][1] + filtered_time_token_list[ii][1]) / 2
                    lower = 0
                else:
                    upper = (filtered_time_token_list[ii + 1][1] + filtered_time_token_list[ii][1]) / 2
                    lower = (filtered_time_token_list[ii - 1][1] + filtered_time_token_list[ii][1]) / 2
                # append phone label, start time, end time tuple
                DCB.append((filtered_time_token_list[ii][0], lower, upper))
            return DCB

        DCB = decision_boundary_calc(filtered_time_token_list, seconds)

        # Assign Maximal labels
        import json
        try:
            with open('str_unic.json') as str_unic_file:
                str_to_unicode_dict = json.loads(str_unic_file.read())
            Max_DCB_init_dict = dict.fromkeys(str_to_unicode_dict, 0)
        except Exception:
            print("Failed to open str_unic.json")

        segVect.insert(0, 0)
        segVect.append(seconds)

        label_list = list()

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

        # Lets zip each label with its start and end times
        segList = list()
        [segList.append((label_list[ii], segVect[ii], segVect[ii + 1])) for ii in range(len(label_list))]

        def clean_segs(segList_in, wav_path):
            segList = segList_in.copy()
            tokens_collapsed = wp.pred_wav_with_collapse(wav_path)

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
                        elif seg_from[0] == transitions[jj][0] and seg_to[0] == transitions[jj][1]:
                            found = True
                            index = index + 1

                        else:
                            index = index + 1
                            break

            return segList

        segList = clean_segs(segList, wav_path)

        temp = list()
        for seg in segList:
            temp.append({"phone": seg[0],
                         "start": seg[1],
                         "stop": seg[2]})
        segList = temp
        return segList

    # Get the tokens
    wp = w2v2_predict.w2v2_predictor()
    #wp.set_model(ckpt="/home/bryce/PycharmProjects/EnsemblePhoneme/wav2vec2-base-timit-demo-phones/checkpoint-11500")
    wp.set_model(ckpt="./wav2vec2-base-timit-demo-phones/checkpoint-11500")
    def forcedAligner(wav_path, vad = True, clean = True, clean_agg = False):
        signalData, samplingFrequency = sf.read(wav_path)

        #Length of speech in seconds
        seconds = len(signalData) / samplingFrequency

        #Get the precollapse tokens
        tokens = wp.pred_wav_no_collapse(wav_path, return_type="phones")

        # Use Unsupseg38 by felix kreuk to get the segmentations
        segVect = seg_pred.pred(wav=wav_path,
                      ckpt='./UnsupSeg38/runs/2021-09-26_16-45-28-default/epoch=4.ckpt',
                      prominence=None)

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
        def decision_boundary_calc(filtered_time_token_list, seconds, bias = 0.5):
            assert 0 <= bias and bias <= 1
            DCB = list()
            for ii in range(len(filtered_time_token_list)):
                if ii == len(filtered_time_token_list) - 1:  # CASE: Last token
                    upper = seconds
                    lower = (filtered_time_token_list[ii - 1][1])*(1-bias) + (filtered_time_token_list[ii][1])*(bias)
                elif ii == 0:  # CASE: First token
                    upper = filtered_time_token_list[ii + 1][1]*(bias) + filtered_time_token_list[ii][1]*(1-bias)
                    lower = 0
                else:
                    upper = (filtered_time_token_list[ii + 1][1])*(bias) + (filtered_time_token_list[ii][1])*(1-bias)
                    lower = (filtered_time_token_list[ii - 1][1])*(1-bias) + (filtered_time_token_list[ii][1])*(bias)
                # append phone label, start time, end time tuple
                DCB.append((filtered_time_token_list[ii][0], lower, upper))
            return DCB

        DCB = decision_boundary_calc(filtered_time_token_list, seconds, bias)

        Max_DCB_init_dict = dict.fromkeys(str_to_unicode_dict, 0)

        #There are comments in the VAD function to use add this there. But you shouldnt.
        segVect.insert(0, 0)
        segVect.append(seconds)

        def maxContribution(segVect, Max_DCB_init_dict, DCB):
            label_list = list()
            #Find the maximum contributer for each segment
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
                        #This should probably be transitions[jj+1][0], not transitions[jj][1]. Too late to change it.
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
                            jj = jj - 1
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
        return segList

    def apply_forcedAligner(batch):
        batch["predict_segs"] = forcedAligner(batch["file"], vad = use_vad, clean = use_clean, clean_agg = clean_aggressive)
        return batch

    filename = experiment_path+"/processed_dataset.pickle"
    if not os.path.isfile(experiment_path+'/processed_dataset.pickle'):
        timit = timit["test"].map(apply_forcedAligner, num_proc = 1)
        timit

        outfile = open(filename, 'wb')
        pickle.dump(timit, outfile)
        outfile.close()
    else:
        infile = open(filename, 'rb')
        timit = pickle.load(infile)
        infile.close()



    def evaluate_results_map(batch):
        hits = 0
        for tseg in batch["target_segs"]:
            for pseg in batch["predict_segs"]:
                if pseg["start"] <= tseg["midpoint"] and tseg["midpoint"] <= pseg["stop"]:
                    pred_list.append({"Predicted":pseg["phone"],"Actual":tseg["phone"]})
                    if tseg["phone"] == pseg["phone"]:
                        hits = hits + 1
                        d_start = tseg["start"] - pseg["start"]
                        d_end = tseg["stop"] - pseg["stop"]
                        time_list.append({'delta_start':d_start, 'delta_end':d_end})

                elif tseg["stop"] < pseg["start"]:
                    break
        length1 = len(batch["target_segs"])
        length2 = len(batch["predict_segs"])
        recall = hits / length1
        acc = hits/length2
        utter_list.append({'recall':recall, 'accuracy':acc, 'hits':hits, 'groundtruth_length':length1, 'prediction_length':length2})

        return batch

    def evaluate_results_AIE_map(batch):
        hits = 0
        tau = (20*10**-3)
        for tseg in batch["target_segs"]:
            for pseg in batch["predict_segs"]:
                if abs(pseg["start"] - tseg["start"]) <= tau:
                    pred_list.append({"Predicted":pseg["phone"],"Actual":tseg["phone"]})
                    if tseg["phone"] == pseg["phone"]:
                        hits = hits + 1
                        d_start = tseg["start"] - pseg["start"]
                        time_list.append({'delta_start':d_start, 'delta_end':0})
                        break

                elif tseg["stop"] < pseg["start"]:
                    break
        length1 = len(batch["target_segs"])
        length2 = len(batch["predict_segs"])
        recall = hits / length1
        acc = hits/length2
        utter_list.append({'recall':recall, 'accuracy':acc, 'hits':hits, 'groundtruth_length':length1, 'prediction_length':length2})

        return batch

    if not os.path.isfile(experiment_path+'/pred_df.pickle') and not os.path.isfile(
            experiment_path+'/time_df.pickle') and not os.path.isfile(experiment_path+'/utter_df.pickle'):


        time_list = list()
        pred_list = list()
        utter_list = list()

        if AIE_evaluation == True:
            timit.map(evaluate_results_AIE_map)
        else:
            timit.map(evaluate_results_map)

        time_df = pd.DataFrame(time_list)
        utter_df = pd.DataFrame(utter_list)
        pred_df = pd.DataFrame(pred_list)

        filename = experiment_path+"/pred_df.pickle"
        outfile = open(filename, 'wb')
        pickle.dump(pred_df, outfile)
        outfile.close()

        filename = experiment_path+"/time_df.pickle"
        outfile = open(filename, 'wb')
        pickle.dump(time_df, outfile)
        outfile.close()

        filename = experiment_path+"/utter_df.pickle"
        outfile = open(filename, 'wb')
        pickle.dump(utter_df, outfile)
        outfile.close()

        pred_df.to_csv(experiment_path+"/pred_df.csv")
        time_df.to_csv(experiment_path+"/time_df.csv")
        utter_df.to_csv(experiment_path+"/utter_df.csv")


    else:
        filename = experiment_path+"/pred_df.pickle"
        infile = open(filename, 'rb')
        pred_df = pickle.load(infile)
        infile.close()

        filename = experiment_path+"/time_df.pickle"
        infile = open(filename, 'rb')
        time_df = pickle.load(infile)
        infile.close()

        filename = experiment_path+"/utter_df.pickle"
        infile = open(filename, 'rb')
        utter_df = pickle.load(infile)
        infile.close()

    print("Mean Accuracy: ", np.mean(utter_df["accuracy"]))

    plt.hist(abs(time_df["delta_start"]*1000),
             range = (0, 150),
             bins = 30,
             density=True,
             color = "skyblue",
             lw=0.1)
    plt.title("Absolute Time Difference of start times of matched and accurate predictions")
    plt.xlabel("Absolute difference in millisecons (ms)")
    plt.ylabel("Density")
    plt.savefig(experiment_path+"/starttimes")
    plt.show()

    plt.hist(abs(time_df["delta_end"]*1000),
             range = (0, 150),
             bins = 30,
             density=True,
             color = "skyblue",
             lw=1)
    plt.title("Absolute Time Difference of end times of matched and accurate predictions")
    plt.xlabel("Absolute difference in millisecons (ms)")
    plt.ylabel("Density")
    plt.savefig(experiment_path + "/endtimes")
    plt.show()

    conf_matrix = metrics.confusion_matrix(pred_df["Actual"], pred_df["Predicted"], labels=list(str_to_unicode_dict.keys()))
    df_cm = pd.DataFrame(conf_matrix, index=list(str_to_unicode_dict.keys()),
                         columns=list(str_to_unicode_dict.keys()))
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm,
               annot=False,
               linewidths=0)
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Confusion matrix of matched predictions")

    print("")
    plt.savefig(experiment_path + "/confusion_matrix")
    plt.show()



    original_stdout = sys.stdout
    with open(experiment_path+'/metrics.txt', 'w') as f:
        sys.stdout = f
        print("Experiment Name:", experiment_name)
        print("Use VAD:", use_vad)
        print("Use clean:", use_clean)
        print("Clean Aggressively:", clean_aggressive)
        print("Bias:", bias)
        print("Eval. method: (AIE: true, SLP: false)", AIE_evaluation)



        print("Start times")
        print("Proportion less than 20ms:", len(time_df[time_df["delta_start"] <= 20/1000])/len(time_df))
        print("Proportion less than 40ms:", len(time_df[time_df["delta_start"] <= 40/1000])/len(time_df))
        print("Proportion less than 60ms:", len(time_df[time_df["delta_start"] <= 60/1000])/len(time_df))

        print("End times")
        print("Proportion less than 20ms:", len(time_df[time_df["delta_end"] <= 20 / 1000]) / len(time_df))
        print("Proportion less than 40ms:", len(time_df[time_df["delta_end"] <= 40 / 1000]) / len(time_df))
        print("Proportion less than 60ms:", len(time_df[time_df["delta_end"] <= 60 / 1000]) / len(time_df))
        print("")

        preds = 0
        for utter in timit["predict_segs"]:
            preds = preds + len(utter)

        msegs = 0 #Verification script
        for ii in timit["phonetic_detail"]:
            msegs = msegs + len(ii["utterance"])
        msegs

        npreds = sum(utter_df['prediction_length'])
        nmanualsegs= sum(utter_df['groundtruth_length'])
        test_df = pred_df[(pred_df['Predicted'] == pred_df['Actual'])]
        correctmatches = len(test_df)
        #Realistically, this accuracy is poorly named. While it is accuracy, its more like the precision
        # i.e. when I predict, what proportion am I getting right?
        accuracy = correctmatches / npreds
        recall = correctmatches / nmanualsegs
        print("Mean proportion of predictions correctly classified: ", np.mean(utter_df["accuracy"]))
        print("Mean proportion of ground truth correctly classified: ", np.mean(utter_df["recall"]))
        print("")
        print("Overall proportion of predictions correctly classified (Nhits / Npreds)", accuracy)
        print("Overall proportion of ground truth correctly classified (Nhits / Nmanualsegs)", recall)
        print("")
        print("Accuracy of predictions (%-Match):", len(test_df)/preds)
        hmean = (((recall)**-1+(accuracy)**-1)/2)**-1
        print("Harmonic Mean Acc:", hmean)

        sys.stdout = original_stdout

        lst = list()
        lst.append({"Bias": 0.25, "hMean": 0.704, "start20": 0.759, "end20": 0.809})
        lst.append({"Bias": 0.3, "hMean": 0.718, "start20": 0.772, "end20": 0.813})
        lst.append({"Bias": 0.35, "hMean": 0.728, "start20": 0.787, "end20": 0.819})
        lst.append({"Bias": 0.4, "hMean": 0.736, "start20": 0.803, "end20": 0.828})
        lst.append({"Bias": 0.45, "hMean": 0.733, "start20": 0.817, "end20": 0.837})
        lst.append({"Bias": 0.50, "hMean":0.723, "start20":0.833, "end20":0.847})
        lst.append({"Bias": 0.55, "hMean": 0.706, "start20": 0.846, "end20": 0.857})
        lst.append({"Bias": 0.60, "hMean": 0.679, "start20": 0.858, "end20": 0.867})
        lst.append({"Bias": 0.65, "hMean": 0.644, "start20": 0.869, "end20": 0.876})
        lst.append({"Bias": 0.70, "hMean": 0.602, "start20": 0.879, "end20": 0.886})
        graph_df = pd.DataFrame(lst)
        graph_df = graph_df.rename(columns={"hMean":"Harmonic Mean", "start20":'Starttime % under 20ms', "end20":'Endtime % under 20ms'})
        graph_df = pd.melt(graph_df, id_vars=['Bias'], value_vars=['Harmonic Mean', 'Starttime % under 20ms', 'Endtime % under 20ms'])
        sn.lineplot(data=graph_df, x="Bias", y="value", hue='variable', markers=True, palette = "viridis")
        plt.title("Harmonic Mean, Start-times and End-times agianst Bias")
        plt.show()








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
        #Condtion: we just instantiated the array
            tmp = arr[jj][0]

        #Condition: rising edge
        elif arr[jj][0] != tmp and tmp == 0:
            #Store timing of rising edge in seconds
            tmp2 = jj * st/len(arr)

            tmp = 1

        # Condition: falling edge
        elif arr[jj][0] != tmp and tmp == 1:
            #add the pair of boundaries to the return list
            return_array.append((tmp2, jj * st/len(arr)))
            tmp = 0

    return return_array



#   removes segmentations which are deemed to have
#   occured in spaces of audio where there is no speech
#
#   Signal:     signal data to process
#   sr:         sampling rate
#   tolerance: tolerance in difference in VAD boundaries and Seg boundaries difference
def filterSegmentations(segmentations, signal, sr, tolerance = 0.05):
    vad=VAD(signal, sr, nFFT=2048, win_length=0.025, hop_length=0.01, theshold=0.5)
    vad = vad.astype('int')
    vad = vad.tolist()
    vadEdges = detectEdges(vad, len(signal)/sr)

    filtered_segs = list()
    # Scan through all the segmentations looking for segments which fit withing the boundaries.
    # Works in O(N^2) time because im a pig.
    #filtered_segs.append(0)
    for seg in segmentations:
        for vadBound in vadEdges:
            if vadBound[0]-tolerance <= seg and seg <= vadBound[1]+tolerance:
                filtered_segs.append(seg)
    #filtered_segs.append(len(signal)/sr)

    return filtered_segs

def filterSegmentationsWrapper(wav_path, segmentations):
    signal, sr  = sf.read(wav_path)
    return filterSegmentations(segmentations, signal, sr)





## Takes :
## wav file
def seg_demo(wav_path):
    signalData, samplingFrequency  = sf.read(wav_path)

    #Duration of utterance in seconds
    seconds = len(signalData)/samplingFrequency

    wp = w2v2_predict.w2v2_predictor()
    wp.set_model(ckpt="/home/bryce/PycharmProjects/EnsemblePhoneme/wav2vec2-base-timit-demo-phones/checkpoint-11500")

    #tokens = ['h#', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'hh', '[PAD]', '[PAD]', '[PAD]', 'eh', '[PAD]', '[PAD]', 'l', '[PAD]', '[PAD]', 'ow', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'pau', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'dh', 'dh', '[PAD]', '[PAD]', 'ih', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 's', '[PAD]', '[PAD]', 'q', '[PAD]', '[PAD]', 'ih', '[PAD]', '[PAD]', '[PAD]', 'z', '[PAD]', '[PAD]', 'ix', '[PAD]', '[PAD]', 'tcl', 'tcl', 't', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'eh', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 's', '[PAD]', '[PAD]', 'tcl', '[PAD]', 't', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 'h#']
    tokens = wp.pred_wav_no_collapse(wav_path, return_type="phones")

    #Use Unsupseg38 by felix kreuk to get the segmentations
    segVect = filterSegmentationsWrapper(wav_path=wav_path,
                           segmentations=seg_pred.pred(wav=wav_path, ckpt='/home/bryce/PycharmProjects/EnsemblePhoneme/UnsupSeg38/runs/2021-09-26_16-45-28-default/epoch=4.ckpt', prominence=None))

    #Delta s is half the distance in time between each token
    delta_s = seconds / (2*len(tokens))

    #A list of tokens with time attached. It's called votelist because itll do some voting later on
    timed_token_list = list()

    #instantiate timestamp with one delta s. The distance between each token in time is 2 times delta_s
    timestamp = delta_s

    #This for loop creats a list of tuples with the timing attached to each
    for token in tokens:
        #Timed token is a tuple with the time in the sequence at which it occurs
        timed_token = (token, timestamp)

        #Add timed token to the voter list
        timed_token_list.append(timed_token)

        #Increment the timestamp for the next token
        timestamp = timestamp + 2*delta_s

    # Now keep only the labels worth interpreting
    filtered_time_token_list = list()
    for tt in timed_token_list:
        if tt[0] != ("[PAD]" or "[UNK]" or "|"):
            filtered_time_token_list.append(tt)

    # Compute Decision Boundaries
    def decision_boundary_calc(filtered_time_token_list, seconds, bias=0.4):
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

    DCB = decision_boundary_calc(filtered_time_token_list, seconds)

    #Assign Maximal labels
    import json
    try:
        with open('str_unic.json') as str_unic_file:
            str_to_unicode_dict = json.loads(str_unic_file.read())
        Max_DCB_init_dict = dict.fromkeys(str_to_unicode_dict, 0)
    except Exception:
        print("Failed to open str_unic.json")

    segVect.insert(0, 0)
    segVect.append(seconds)

    label_list = list()

    for segIndex in range(len(segVect)):
        label_dict = Max_DCB_init_dict.copy()

        if segIndex != (len(segVect) - 1):
            t_segStart = segVect[segIndex]
            t_segEnd = segVect[segIndex+1]
            for dcb in DCB:
                #CASE: decision starts within the segment
                if  t_segStart <= dcb[1] and dcb[1] <= t_segEnd:
                    #CASE: Decision contained entirely within the segment
                    if dcb[2] <= t_segEnd :
                        label_dict[dcb[0]] = label_dict[dcb[0]] + (dcb[2] - dcb[1])
                    else: #CASE: Decision starts within the segment but ends elsewhere
                        label_dict[dcb[0]] = label_dict[dcb[0]] + (t_segEnd - dcb[1])
                #CASE: Decision ends within the segment, but does not start within the segmnet
                elif t_segStart  <= dcb[2] and dcb[2] <= t_segEnd:
                    label_dict[dcb[0]] = label_dict[dcb[0]] + (dcb[2] - t_segStart)
                #CASE: Decision contains the entirety of the seg
                elif  dcb[1] <=  t_segStart and t_segEnd <= dcb[2] :
                    label_dict[dcb[0]] = label_dict[dcb[0]] + (t_segEnd - t_segStart)
                else:
                    pass
                #dcb[0] : phone label. dcb[1] : start time, dcb[2] : end time

            label_list.append(max(label_dict, key=label_dict.get))

    #Lets zip each label with its start and end times
    segList = list()
    [segList.append((label_list[ii], segVect[ii], segVect[ii+1])) for ii in range(len(label_list))]

    def clean_segs(segList_in, wav_path):
        segList = segList_in.copy()
        tokens_collapsed = wp.pred_wav_with_collapse(wav_path)

        transitions = list()
        for ii in range(len(tokens_collapsed)-1):
            transitions.append((tokens_collapsed[ii], tokens_collapsed[ii+1]))

        index = 0
        for jj in range(len(transitions)):
            found = False
            limitreached = False

            while found == False and limitreached == False:
                if index >= len(segList)-1:
                    limitreached = True
                else:
                    seg_from = segList[index]
                    seg_to = segList[index + 1]

                    #CASE: the two elements are the same, ie seglist: aab, transition: ab, focal:aa, turn seglist into ab
                    if seg_from[0] == seg_to[0] and seg_from[0] == transitions[jj][0]:
                        segList[index] = (segList[index][0], segList[index][1], segList[index+1][2])
                        segList.remove(segList[index+1])


                    #CASE: Transition is found
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
        transitions = list()
        for ii in range(len(tokens_collapsed)-1):
            transitions.append((tokens_collapsed[ii], tokens_collapsed[ii+1]))

        ceiling = len(segList)-2
        jj=0
        finished = False
        while finished == False:
            if jj <= ceiling:
                if segList[jj][0]==segList[jj+1][0]:
                    if not (segList[jj][0],segList[jj+1][0]) in transitions:
                        newSeg = (segList[jj][0], segList[jj][1], segList[jj+1][2])
                        segList[jj] = newSeg
                        segList.remove(segList[jj+1])
                        ceiling = ceiling-1
                    jj = jj - 1
                jj = jj + 1
            else:
                finished = True

        return segList
    segList = clean_segs_aggressive(segList, wav_path)

    temp = list()
    for seg in segList:
        temp.append({"phone":seg[0],
                     "start":seg[1],
                     "stop":seg[2]})
    segList = temp
    return segList

## Checks to see if you have the model binaries downloaded.
## If not - grabs a copy from cloudstor
def model_from_cloudstor():
    #w2v2ckptdir = "wav2vec2-base-timit-demo-phones"
    w2v2ckptdir = "wav2vec2-base-timit-demo-phones"
    if os.path.isdir(w2v2ckptdir):
        print("w2v2 model folder found!")
    else:
        download_yn = input("W2V2 model not found (looking for directory ", w2v2ckptdir,") - would you like to install "
                                                                                        "from cloudstor? [y/n]")

        if 'y' in download_yn:
            os.mkdir(w2v2ckptdir)
            if not os.path.isfile("w2v2-sample-model.tar.gz"):
                print("w2v2 model was not found. obtaining a checkpoint from cloudstor...")
                print("Approx 3.5GB - this may take a while - please wait :)")

                # https://github.com/underworldcode/cloudstor/blob/master/cloudstor/Examples/Notebooks/CloudstorFiles.ipynb
                public_file = cloudstor(url="82YfuY0nnNGMUAY", password='')
                print(public_file.list())
                public_file.download_file("w2v2-sample-model.tar (2).gz", "w2v2-sample-model.tar.gz")

            # https://www.geeksforgeeks.org/how-to-uncompress-a-tar-gz-file-using-python/
            #tarball = tarfile.open('w2v2-sample-model.tar.gz')
            #tarball.extractall(w2v2ckptdir)
            #tarball.close()

            shutil.unpack_archive('w2v2-sample-model.tar.gz', w2v2ckptdir)
        else:
            print("\'no\' input received. You may manually download a copy from https://cloudstor.aarnet.edu.au/plus/s/82YfuY0nnNGMUAY")


if __name__ == "__main__":
    #Perform directory and model check
    model_from_cloudstor()

    #This is the demo function
    utilities.FA_demo("./helloworld.wav")

    #use main to evaluate or perform an experiment
    #main()