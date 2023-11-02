import numpy as np
import FlowCal as FC
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import svm
import random
import pickle

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#########################################################
# The samples in the dataset consist of vary length. 
# The approach that is implamented here is by spliting the sample into several windows with fixed length, 
# hence we can create a model with fixed number of features.
#########################################################

def data_loading():
    # Load the Marker Channel Mapping
    EU_marker_channel_mapping = pd.read_excel('EU_marker_channel_mapping.xlsx')

    use_channel = []
    # List the channel that can be used
    for i in range(len(EU_marker_channel_mapping.index)):
        if EU_marker_channel_mapping['use'][i]==1:
            use_channel.append(EU_marker_channel_mapping['PxN(channel)'][i])

    data = {}
    flag = 0

    # Load the FCS Data
    for folder in os.listdir('raw_fcs'):
        data_dir = os.path.join('raw_fcs', folder)
        data_dir = os.path.join(data_dir, os.listdir(data_dir)[0])
        data_temp = FC.io.FCSData(data_dir)
        
        # List the unused channel
        if flag==0:
            flag = 1
            data_copy = data_temp.copy()
            unused_channel = []
            for i, channels_ in enumerate(data_copy.channels):
                if channels_ not in use_channel:
                    unused_channel.append(channels_)
            
        unused_index = []
        for i in range(len(data_temp.channels)):
            if data_temp.channels[i] in unused_channel:
                unused_index.append(i)

        # Transform the FSC data to array and delete the unused channel
        data[folder] = np.delete(np.asarray(data_temp), unused_index, axis=1)

    return data

def data_preparation(data, window_size):
    EU_label = pd.read_excel('EU_label.xlsx')
    label_filename_list = list(EU_label['file_flow_id'])
    
    # Split the training and testing data with 8:2 ratio. 8 for training and 2 for testing
    data_keys = list(data)
    random.shuffle(data_keys)
    train_test_split_ratio = 0.8
    train_data_keys = data_keys[:int(0.8*len(data_keys))]
    test_data_keys = data_keys[int(0.8*len(data_keys)):]
    
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    
    # Determine the sliding window stride
    stride = int(np.ceil(window_size/2))
    
    # Split the data into window samples
    for i, key in enumerate(train_data_keys):
        sample_len = int(np.ceil(((data[key].shape[0]-window_size)/stride)+1))
        for j in range(0, sample_len):
            
            cropped_data = np.asarray(data[key][stride*j:stride*j+window_size])
            if cropped_data.shape[0] < window_size:
                cropped_data = np.pad(cropped_data, ((0,window_size-cropped_data.shape[0]),(0,0)), 'mean')
            
            train_X.append(cropped_data)
            
            if EU_label['label'][label_filename_list.index(key)]=='Healthy':
                train_Y.append(1)
            else:
                train_Y.append(0)
    
    # Shuffle the training data
    zipped_data = list(zip(train_X, train_Y))
    random.shuffle(zipped_data)
    train_X, train_Y = zip(*zipped_data)
                
    # List the testing data
    for i, key in enumerate(test_data_keys):
        test_X.append(data[key])
        
        if EU_label['label'][label_filename_list.index(key)]=='Healthy':
            test_Y.append(1)
        else:
            test_Y.append(0)
    
    return np.asarray(train_X), np.asarray(train_Y), test_X, test_Y

def train_model(train_X, train_Y):

    train_X = train_X.reshape(len(train_X), -1)

    print(" START TRAINING \n")

    # Initiate the Linear SVM Classification
    clf = make_pipeline(StandardScaler(), LinearSVC(dual=False, random_state=0, tol=1e-5))
    
    # Traing the model
    clf.fit(train_X, train_Y)

    print(" DONE TRAINING \n")
    
    print(" SAVING MODEL PARAMETERS \n")
    with open('SVM_param1.pkl', 'wb') as fp:
        pickle.dump(clf.get_params(), fp)
        
    return clf

def test_model(clf, test_X, test_Y, window_size):
    
    stride = int(np.ceil(window_size/2))
    
    acc = 0
    
    # Testing the model using testing set
    for i in range(len(test_X)):
        test_X_list = []
        test_Y_list = []
        sample_len = int(np.ceil(((test_X[i].shape[0]-window_size)/stride)+1))
        
        for j in range(0, sample_len):
            
            cropped_data = np.asarray(test_X[i][stride*j:stride*j+window_size])
            if cropped_data.shape[0] < window_size:
                cropped_data = np.pad(cropped_data, ((0,window_size-cropped_data.shape[0]),(0,0)), 'mean')

            test_X_list.append(cropped_data)
        
        test_X_sample = np.asarray(test_X_list).reshape(len(test_X_list), -1)
        
        predict = clf.predict(test_X_sample)
        
        # Determine accuracy by finding any 'sick' prediction between the window samples
        if any(predict==0):
            class_predict = 0
        else:
            class_predict = 1
            
        if class_predict==test_Y[i]:
            acc+=1
            
    acc = acc/len(test_Y)
    
    print(" Acc: ", acc)

def main():
    data = data_loading()
    print(" DONE LOADING DATA \n")
    window_size = 1000
    train_X, train_Y, test_X, test_Y = data_preparation(data, window_size)
    print(" DONE PREPARING DATA \n")
    
    clf = train_model(train_X, train_Y)
    
    print(" TESTING MODEL \n")
    test_model(clf, test_X, test_Y, window_size)


main()