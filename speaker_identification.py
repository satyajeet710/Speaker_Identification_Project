
import os
import librosa
import numpy as np

def load_data(dir):
    audio_files=[]
    labels=[]
    print(f"Loading data from directory: {dir}")
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)

                # Check if the file is directly inside the testing_audio
                if os.path.basename(root) == os.path.basename(dir):
                    label = os.path.splitext(file)[0]  #label is file name
                else:
                    label = os.path.basename(root)     #label is parent folder name

                label= label.split('-')[0].split('_')[0]
                audio_files.append(file_path)
                labels.append(label)
    print(f"Total audio files loaded: {len(audio_files)}")
    return audio_files, labels

def mfcc_features(audio_files,n_mfcc=13):
    mfcc_features=[]

    for file in audio_files:
        y, sr = librosa.load(file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_features.append(mfcc_mean)

    return mfcc_features

train_dir = "GMM-Speaker-Identification-data\GMM-Speaker-Identification-data\GMM-Speaker-Identification-data\Speaker_data\Voice_Samples_Training"
test_dir="GMM-Speaker-Identification-data\GMM-Speaker-Identification-data\GMM-Speaker-Identification-data\Speaker_data\Testing_Audio"

# Load audio files and labels
train_files,train_labels= load_data(train_dir)
test_files,test_labels= load_data(test_dir)

# print(len(train_files))

x_train= mfcc_features(train_files)
x_test= mfcc_features(test_files)
x_train=np.array(x_train)
train_labels=np.array(train_labels)

# print(x_train.shape)
print(train_labels.shape)
print(train_labels)
# print(x_test)
# print(len(x_test))
# print(test_labels)

import numpy as np
from sklearn.mixture import GaussianMixture

def gmm_model(X_train, y_train,n_components=8,max_iter=200, reg_covar=1e-1):
    speaker_models = {}
    num_speakers = np.unique(y_train)

    for speaker in num_speakers:
        speaker_features = X_train[y_train == speaker]
        n=speaker_features.shape[0]
        if n< n_components:   #samples are less than components
            n_components=n

        try:
            gmm = GaussianMixture(n_components=n_components, max_iter=max_iter,
                                  covariance_type='diag', reg_covar=reg_covar)
            gmm.fit(speaker_features)  # Use original shape of speaker_features
            speaker_models[speaker] = gmm
            print(f"Trained GMM for speaker: {speaker}")
        except ValueError as e:
            print(f"Failed to train GMM for speaker: {speaker}. Error: {str(e)}")

    return speaker_models

def predict(speaker_models,mfcc_feature):
    log_likelihoods = {}
    mfcc_feature = np.array(mfcc_feature).reshape(1, -1)

    for speaker, model in speaker_models.items():
        log_likelihoods[speaker] = model.score(mfcc_feature)
    pred_speaker = max(log_likelihoods, key=log_likelihoods.get)
    return pred_speaker, log_likelihoods

# Train GMM models
speaker_models = gmm_model(x_train, train_labels)

def accuracy(x,y,speaker_models):
    c=0
    for i in range(len(x)):
        mfcc_feature=x[i]
        actual_speaker=y[i]
        pred_speaker,log_likelihoods=predict(speaker_models, mfcc_feature)
        print(f"{i + 1}: Actual Speaker: {actual_speaker}, Predicted Speaker: {pred_speaker}")
        if pred_speaker==actual_speaker:
            c+= 1
    return c

c_train=accuracy(x_train,train_labels,speaker_models)
c_test=accuracy(x_test,test_labels,speaker_models)

print("accuracy for training data is", (c_train/(len(x_train))*100 + 1e-10))
print("accuracy for testing data is", (c_test/(len(x_test))*100 + 1e-10))