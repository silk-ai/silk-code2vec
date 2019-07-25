import pandas as pd
import numpy as np
import tensorflow as tf
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint

def load_dataset(file_path):
    data = pd.read_csv(file_path, sep=",", header=None)
    return data.to_numpy().flatten()

def get_dict(data):
    name_set = set(data.tolist())
    name_dict = dict(enumerate(name_set))
    return name_dict

def name_to_int(data, dict):
    processed_data = []
    for name in data:
        processed_data.append(list(dict.keys())[list(dict.values()).index(name)])
    return processed_data

def make_windows(data, window_len, num_features):
    network_input = []
    network_output = []
    #create input sequences and their corresponding output sequences
    for i in range(0, len(data) - window_len, num_features):
        network_input.append(data[i:i + window_len])
        network_output.append(data[i + window_len])
    return np.array(network_input), np.array(network_output)

def get_model(weights_file_path, num_classes):
    model = Sequential() #We are creating a sequential model
    model.add(LSTM(256, input_shape=(2, 1), return_sequences=True))
    #model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    #model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    #model.add(Dropout(0.3))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.load_weights(weights_file_path)
    return model

def generate_prediction(input_list, model, dict):
    num_classes = len(dict)
    pattern = name_to_int([input_list[0], input_list[1]], dict)
    prediction_output = [input_list[0], input_list[1]]
    for name in range(len(input_list)-3):
        prediction_input = np.reshape(pattern, (1, 2, 1))
        prediction_input = prediction_input / float(num_classes)
        prediction = model.predict(prediction_input, verbose=0)
        prediction = list(prediction).sort(reverse=True)
        for pred in  prediction:
            if pred in input_list:
                prediction_output.append(dict[pred])
                input_list.remove(dict[pred])
                pattern = np.append(pattern,pred)
                pattern = pattern[1:len(pattern)]
                break
    return prediction_output

def do(input_list, data_file_path, weights_file_path):
    data = load_dataset(data_file_path)
    dict = get_dict(data)
    model = get_model(weights_file_path, len(dict))
    recommendation = generate_prediction(input_list, model, dict)
    return recommendation