import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import sys

from keras.models import Model
from os.path import isfile
from keras.layers import *

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def reduce(dataset_path, output_file, number_of_timeseries = None, window_length = 10, epochs = 75, test_samples = 100, graphs_to_show = 0):
  nasdaqDF = pd.read_csv(dataset_path,sep="\t", header=None)

  if number_of_timeseries == None:
    number_of_timeseries = len(nasdaqDF)

  stock_values = nasdaqDF.iloc[:,1:]
  stock_names = nasdaqDF.iloc[:,0]


  input_window = Input(shape=(window_length,1))

  x = Conv1D(8, 3, activation="relu", padding="same")(input_window) # 10 dims
  x = MaxPooling1D(2,padding="same")(x) # 5 dims
  x = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
  encoded = MaxPooling1D(2, padding="same")(x) # 3 dims

  encoder = Model(input_window, encoded)

  # 3 dimensions in the encoded layer

  x = Conv1D(1, 3, activation="relu", padding="same", name = 'bottleneck_layer')(encoded) # 3 dims
  x = UpSampling1D(2)(x) # 6 dims
  x = Conv1D(8, 2, activation='relu')(x) # 5 dims
  x = UpSampling1D(2)(x) # 10 dims
  decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims

  autoencoder = Model(input_window, decoded)
  autoencoder.summary()
  autoencoder.compile(loss='binary_crossentropy')

  def takeX(numpyArray, window_length):  
    X_ = np.reshape( numpyArray, (numpyArray.shape[0]*numpyArray.shape[1]) )
    return X_

  def findNameFromIndex(stock_names, index):
    try: 
      return stock_names.loc[index]
    except:
      print('index=', index, ' out of bounds!')

  reduced_dict = {}
  for stock_index in range(number_of_timeseries):

    df = pd.DataFrame(np.array(stock_values.iloc[[stock_index]].T), columns=['price'])
    prices = np.array(df.price)
    prices = np.reshape(prices, (prices.shape[0], 1))
    num_windows = math.floor(len(df['price'])/window_length)

    scale_dict = {}
    X_complete = []
    for i in range(num_windows):
      scale_dict[i] = MinMaxScaler()
      X_complete.append(scale_dict[i].fit_transform(df['price'].values[i*window_length:(i+1)*window_length].reshape(-1, 1)))
    X_complete = np.array(X_complete)

    X_test = X_complete[-test_samples:]
    X_train = X_complete[:-test_samples]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    history = autoencoder.fit(X_train, X_train,
                  epochs=epochs,
                  batch_size=512,
                  shuffle=False,
                  validation_data=(X_test, X_test))
    
    reduced_stock = encoder.predict(X_complete)
    reduced_stock = np.reshape([scale_dict[i].inverse_transform(reduced_stock[i]) for i in range(num_windows)], (-1,1))
    reduced_stock = takeX(reduced_stock, 3)
    stock_name = findNameFromIndex(stock_names,stock_index)
    reduced_dict[stock_name] = reduced_stock

  df_reduced = pd.DataFrame.from_dict(reduced_dict , orient='index')
  df_reduced.to_csv(output_file,sep="\t", header=None)

  for stock_index in range(graphs_to_show):
    stock_ = findNameFromIndex(stock_names,stock_index)

    plt.figure(figsize=(25, 5))
    plt.plot(reduced_dict[stock_], color='red', marker='*' )
    plt.show()

    plt.figure(figsize=(25, 5))
    plt.plot(nasdaqDF.loc[[stock_index]].values.tolist()[0][1:], color='teal', marker='*')
    plt.show()

#reding command line arguments
i = 1
data_input_path = None
data_output_file = None
query_input_path = None
query_output_path = None
while i <len(sys.argv):
  if sys.argv[i].strip().lower() == '-d':
    i += 1
    data_input_path = sys.argv[i]
  if sys.argv[i].strip().lower() == '-od':
    i += 1
    data_output_file = sys.argv[i]
  if sys.argv[i].strip().lower() == '-q':
    i += 1
    query_input_path = sys.argv[i]
  elif sys.argv[i].strip().lower() == '-oq':
    i += 1
    query_output_path = sys.argv[i]
  i += 1

if data_input_path == None or data_output_file == None or query_input_path == None or query_output_path == None:
  print('reduce.py: not enough arguments, check usage!')
  exit(-1)


reduce(data_input_path,data_output_file,5,graphs_to_show=3)
reduce(query_input_path,query_output_path,5,graphs_to_show=3)