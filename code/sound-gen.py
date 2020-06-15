import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt


from scipy.io import wavfile
sample_rate, wav_data = wavfile.read("Lambo-rev-sound.wav")
wav_data = wav_data[10000:,0]


maxlen = 100
step = 1
x = []
y = []
for i in range(0, (wav_data.shape[0] - maxlen), step):
    x.append(wav_data[i: i + maxlen])
    y.append(wav_data[i + maxlen])
print('nb sequences:', len(x))

x = np.asarray(x)
x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

y = np.asarray(y)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

model = Sequential()
model.add(LSTM(64, input_shape=(1, 100), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
optimizer = Adam()
model.compile(loss=root_mean_squared_error, optimizer=optimizer)


model.fit(x, y,
          batch_size=256,
          epochs=30)

# To create a 4 second output, below for loop
output = []
for i in range(44100*3, 44100*7):
    x_pred = x[i]
    x_pred = np.reshape(x_pred, (1, 1, x_pred.shape[1]))

    pred = model.predict(x_pred, verbose=0)[0]
    output.append(pred)


output = np.asarray(output)

fs = 44100
out_f = 'output.wav'

wavfile.write(out_f, fs, output)