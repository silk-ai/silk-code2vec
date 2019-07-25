# Load dataset
#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint

#%%
def load_dataset(file_path):
    data = pd.read_csv(file_path, sep=",", header=None)
    return data.to_numpy().flatten()

#%%
raw_data = load_dataset("/home/edge/Desktop/Hackathon/silk-code2vec/generatorRNN/silk_dataset.txt")

#%%
name_set = set(raw_data.tolist())
name_dict = dict(enumerate(name_set))
num_classes = len(name_set)
print(name_dict)

#%%
processed_data = []
for name in raw_data:
    processed_data.append(list(name_dict.keys())[list(name_dict.values()).index(name)])

#%%
def make_windows(data, window_len, num_features):
    network_input = []
    network_output = []
    #create input sequences and their corresponding output sequences
    for i in range(0, len(data) - window_len, num_features):
        network_input.append(data[i:i + window_len])
        network_output.append(data[i + window_len])
    
    return np.array(network_input), np.array(network_output)
#%%
WINDOW_LEN = 2
DATA_LEN = len(raw_data)
NUM_FEATURES = 1

rnn_input, rnn_output = make_windows(processed_data, WINDOW_LEN, NUM_FEATURES)

num_patterns = len(rnn_input) #number of example sequences

#%%
# norma#lize input to values 0<= x <= 1
normalized_rnn_input = rnn_input / float(num_classes)
normalized_rnn_input = normalized_rnn_input.reshape(normalized_rnn_input.shape[0], normalized_rnn_input.shape[1], 1)
# One hot encoding the output
encoded_rnn_output = np_utils.to_categorical(rnn_output) 

#%%
print(normalized_rnn_input.shape)
#%%
model = Sequential() #We are creating a sequential model
model.add(LSTM(256, input_shape=(normalized_rnn_input.shape[1], normalized_rnn_input.shape[2]), return_sequences=True))
#model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
#model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
#model.add(Dropout(0.3))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print(model.summary())

#%%
filepath = "saved_weights/rnn_weights-{epoch:02d}-{loss:.4f}-bigger.hdf5"    
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
)    

callbacks_list = [checkpoint]     

#%%
model.fit(normalized_rnn_input, encoded_rnn_output, epochs=20, 
          batch_size=128, shuffle=True, callbacks=callbacks_list)


#%%
model.load_weights('saved_weights/rnn_weights-20-5.2481-bigger.hdf5')


#%%
start = np.random.randint(0, len(normalized_rnn_input)-1)
pattern = normalized_rnn_input[start]
prediction_output = []
# generate 500 notes
for note_index in range(100):
    prediction_input = np.reshape(pattern, (1, 2, 1))
    prediction_input = prediction_input / float(num_classes)
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = name_dict[index]
    prediction_output.append(result)
    pattern = np.append(pattern, index)
    pattern = pattern[1:len(pattern)]

#%%
print(prediction_output)

#%%
