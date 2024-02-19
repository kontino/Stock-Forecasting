import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

from keras.models import Sequential
from keras.models import load_model
from os.path import isdir
from keras.layers import *

from sklearn.preprocessing import StandardScaler

number_of_total_stocks_to_be_used = 10

def our_train_test_split(dataframe, percentage):
  if percentage > 1:
    return (None,None)
  else:
    split_index = int(percentage*dataframe.shape[0])
    return ( dataframe.iloc[: split_index , :], dataframe.iloc[split_index :, : ] )

def getXy(numpyArray, range_start, range_end):
  print(numpyArray.shape)
  if range_start > range_end:
    return (None,None)
  X_ = []
  y_ = []
  for j in range(numpyArray.shape[1]):
    X_.append([])
    y_.append([])
    for i in range(range_start, range_end):
        X_[j].append(numpyArray[i-range_start:i, j])
        y_[j].append(numpyArray[i, j])
  print("X ", len(X_), "X[]", len(X_[0]))

  for j in range(numpyArray.shape[1]):
    X_[j] = np.array(X_[j])
    y_[j] = np.array(y_[j])
    X_[j] = np.reshape(X_[j], (X_[j].shape[0], X_[j].shape[1], 1))
  return (X_, y_)

def findIndexFromName(stock_names, name):
  try: 
    return (stock_names.loc[stock_names == name]).index[0]
  except:
    print('name: ', name, ' does not exist in dataframe!')

def findNameFromIndex(stock_names, index):
  try: 
    return stock_names.loc[index]
  except:
    print('index=', index, ' out of bounds!')

def LSTMAutoencodingTrain(X_train, y_train, epochs = 5):
  model = Sequential()
  model.add(LSTM(
    units=64,
    input_shape=(X_train[0].shape[1], X_train[0].shape[2])
  ))
  model.add(Dropout(rate=0.2))
  model.add(RepeatVector(n=X_train[0].shape[1]))
  model.add(LSTM(units=64, return_sequences=True))
  model.add(Dropout(rate=0.2))
  model.add(
  TimeDistributed(Dense(units=X_train[0].shape[2])))
  model.compile(loss='mae', optimizer='adam')

  for i in range(len(X_train)):
    print("Fitting stock #", i)
    model.fit(
      X_train[i], y_train[i],
      epochs=epochs,
      batch_size=32,
      validation_split=0.1,
      shuffle=False
    )
  return model

class Autoencoder:
  def __init__(self, input_file, input_file_seperator = '\t', saved_model = None):
    self.inputDF = pd.read_csv(input_file, sep=input_file_seperator, header=None).head(number_of_total_stocks_to_be_used)
    self.stock_values = self.inputDF.iloc[:,1:]
    self.stock_names = self.inputDF.iloc[:,0]
    if saved_model != None:
      self.multimodel = load_model(saved_model)
    else:
      self.multimodel = None

  def encode(self, train_percentage, epoch, time_steps = 25, threshold = 0.45, stock_name = None, stock_index = None, save_model = None):
    if stock_index == None and stock_name == None :
      return None
    elif stock_index == None:
      stock_index = findIndexFromName(self.stock_names, stock_name)
    elif stock_name == None:
      stock_name = findNameFromIndex(self.stock_names, stock_index)
    else:
      print('predictStock: Error both name and index given!')
      return None


    #scaling
    scaler = StandardScaler()
    linscaler = StandardScaler()

    # scaler.fit(self.stock_values.T)

    scaled_Values = scaler.fit_transform(self.stock_values.T)
    scaled_ValuesDF = pd.DataFrame(scaled_Values)
    testScaledValues = linscaler.fit_transform(self.stock_values.iloc[[stock_index]].T)
    testScaledValuesDF = pd.DataFrame(testScaledValues)

    #rescale
    rescaled_values = scaler.inverse_transform(scaled_Values)
    rescaled_valuesDF = pd.DataFrame(rescaled_values)

    #spliting train and test data
    df_train, df_test = our_train_test_split(scaled_ValuesDF, train_percentage)
    print('train size', df_train.shape, 'test size', df_test.shape)

    #extracting X,y train
    X_train, y_train = getXy( df_train.to_numpy(), time_steps, df_train.shape[0] )
    # print('x train size', X_train.shape, 'y train size', y_train.shape)

    #extracting X test
    X_test, y_test = getXy( df_test.to_numpy(), time_steps, df_test.shape[0] )


    _, df_test = our_train_test_split(testScaledValuesDF, train_percentage)

    # print('x test size', X_test.shape)

    if self.multimodel == None:
      model = LSTMAutoencodingTrain(X_train, y_train, epoch)
      self.multimodel = model
    else:
      model = self.multimodel

    if save_model:
        model.save(save_model)

    predicted_stock_price = model.predict(X_train[stock_index])
    print(predicted_stock_price.shape)
    print(X_train[stock_index].shape)
    #calculating train loss
    train_mae_loss = np.mean(np.abs(predicted_stock_price - X_train[stock_index]), axis=1)

    X_test_pred = model.predict(X_test[stock_index])
    #calculating test loss
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test[stock_index]), axis=1)

    test_score_df = pd.DataFrame(index=df_test[time_steps:].index)
    
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['value'] = df_test.iloc[time_steps:]
    test_score_df['date'] = df_test[time_steps:].index
    

    anomalies = test_score_df.loc[ test_score_df['anomaly'] == True ]

    plt.figure(figsize=(20, 10))
    sns.lineplot(x=test_score_df['date'], y=test_score_df['value'] )
    sns.scatterplot(x=test_score_df['date'], y=anomalies['value'], color = 'red')
    plt.show()

#reding command line arguments
i = 1
dataset_path = None
number_of_timeseries = None
error_value = None
saved_model_arg = None
while i <len(sys.argv):
  if sys.argv[i].strip().lower() == '-d':
    i += 1
    dataset_path = sys.argv[i]
  elif sys.argv[i].strip().lower() == '-n':
    i += 1
    number_of_timeseries = int(sys.argv[i])
  elif sys.argv[i].strip().lower() == '-mae':
    i += 1
    error_value = float(sys.argv[i])
  elif sys.argv[i].strip().lower() == '--pretrained':
    i += 1
    saved_model_arg = sys.argv[i]
  i += 1

if dataset_path == None or number_of_timeseries == None or error_value == None:
  print('detect.py: not enough arguments, check usage!')
  exit(-1)

if saved_model_arg != None:
  if isdir(saved_model_arg) == False:
    print('detect.py: ' + saved_model_arg + ' there is no such folder, pre trained model will not be used!')

ae = Autoencoder(dataset_path, saved_model = saved_model_arg)

for i in range(0,number_of_timeseries):
  ae.encode(train_percentage = 0.7, epoch = 2, time_steps = 25, threshold = error_value, stock_index = i, save_model = None)