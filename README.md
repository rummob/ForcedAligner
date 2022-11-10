# Read me
Author: Bryce Wohlan

Text independent forced aligner tool developed as part of my 2021 electrical engineering honor's thesis at Curtin University.

Supervisors: 
Duc Son (Sonny) Pham
Kit Yan Chan


# Code Reuse Acknowledgements
**Felix Kreuk** -  Author of Self-Supervised Contrastive Learning for Unsupervised Phoneme Segmentation (INTERSPEECH 2020) (Unsup Seg) - https://github.com/felixkreuk/UnsupSeg - Source code used in the "UnsupSeg38" folder, albeit I updated this to python 3.8

**eesungkim** - Implemented Sohn's Statistical VAD in public repo - https://github.com/eesungkim/Voice_Activity_Detector - code reused in this repo under the utils folder. 

**Patrick Von Platen** - A lot of the wav2vec2 implementation is based off his HuggingFace tutorial - https://github.com/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb - Probably would not have made nearly as much progress as I did without this.

# Requirements
See the provided requirements.txt - or use the following:

    numpy~=1.18.1  
    scipy~=1.6.2  
    matplotlib~=3.4.2  
    pandas~=1.3.1  
    seaborn~=0.11.2  
    soundfile~=0.10.3.post1  
    datasets~=1.11.0  
    cloudstor~=0.1.4  
    scikit-learn~=1.0  
    torch~=1.4.0  
    tqdm~=4.42.1  
    wandb~=0.12.2  
    dill~=0.3.1.1  
    torchaudio~=0.4.0  
    librosa~=0.7.2  
    boltons~=20.0.0  
    transformers~=4.4.0  
    jiwer~=2.2.0  
    dataclasses~=0.8

# Usage
This repo allows you to perform text-independent alignment on an audio file. Requires a W2V2 model and an Unsupseg model.

W2V2 provides information on the segment labelling - UnsupSeg provides information on the segments. My contribution is a way of coordinating the two models to make a sensible forced aligner.

Uncomment the functionality in main.py under the "if \_\_name\_\_ == \_\_main\_\_" part.

**main()** will perform an experiment.
**utilities.FA_demo(wav_file_path)** will perform segmentation on the given raw wav file (must be single channel, 16000hz sampling rate)
**model_from_cloudstor()** will collect a w2v2 model from cloudstor for you if you don't already have one - this couldn't be provided in this repo as it greater than 2GB (and I don't feel like coughing up the coin for Github team)

