import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

from keras.models import Sequential
from keras.models import load_model
from os.path import isdir
from keras.layers import *

from sklearn.preprocessing import MinMaxScaler

number_of_total_stocks_to_be_used = 10

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

def LSTMtrain(X_train, y_train, epochs = 5):
  model = Sequential()

  #Adding the first LSTM layer and some Dropout regularisation
  model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train[0].shape[1], 1)))
  model.add(Dropout(0.2))
  # Adding a second LSTM layer and some Dropout regularisation
  model.add(LSTM(units = 50, return_sequences = True))
  model.add(Dropout(0.2))
  # Adding a third LSTM layer and some Dropout regularisation
  model.add(LSTM(units = 50, return_sequences = True))
  model.add(Dropout(0.2))
  # Adding a fourth LSTM layer and some Dropout regularisation
  model.add(LSTM(units = 50))
  model.add(Dropout(0.2))
  # Adding the output layer
  model.add(Dense(units = 1))

  # Compiling the RNN
  model.compile(optimizer = 'adam', loss = 'mean_absolute_error')

  # Fitting the RNN to the Training set
  for i in range(len(X_train)):
    model.fit(X_train[i], y_train[i], epochs = epochs, batch_size = 32)
  return model

class StockPredictor:
  def __init__(self, input_file, input_file_seperator = '\t', saved_multi_model = None):
    self.inputDF = pd.read_csv(input_file, sep=input_file_seperator, header=None)
    self.inputDF = self.inputDF.head(number_of_total_stocks_to_be_used)
    self.stock_values = self.inputDF.iloc[:,1:]
    self.stock_names = self.inputDF.iloc[:,0]

    if saved_multi_model != None:
      self.multimodel = load_model(saved_multi_model)
    else:
      self.multimodel = None

  def predictUni(self, train_percentage, epoch, stock_name = None, stock_index = None, saved_model = None, save_model = None):
    
    #scaling
    scaler = MinMaxScaler( feature_range = (0, 1) )
    stock_values_scaled_transposed = scaler.fit_transform(self.stock_values.iloc[[stock_index]].T)
    stock_values_scaled_transposedDF = pd.DataFrame(stock_values_scaled_transposed)

    #rescale
    rescaled_values = scaler.inverse_transform(stock_values_scaled_transposedDF)
    rescaled_valuesDF = pd.DataFrame(rescaled_values)

    #spliting train and test data
    df_train, df_test = our_train_test_split(stock_values_scaled_transposedDF, train_percentage)
    print('train size', df_train.shape, 'test size', df_test.shape)
    df_train_orig, df_test_orig = our_train_test_split(rescaled_valuesDF, train_percentage)

    #extracting X,y train
    X_train, y_train = getXy( df_train.to_numpy(), 25, df_train.shape[0] )
    X_train, y_train = X_train[0], y_train[0] #100 iq play
    print('x train size', X_train.shape, 'y train size', y_train.shape)

    #extracting X test
    X_test, y_test = getXy( df_test.to_numpy(), 25, df_test.shape[0] )
    X_test, y_test = X_test[0], y_test[0]
    print('x test size', X_test.shape)


    if saved_model == None:
      model = LSTMtrain([X_train], [y_train], epoch)
    else:
      model = load_model(saved_model)

    if save_model != None:
        model.save(save_model)

    #Rescaling
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    predicted_stock_priceDF = pd.DataFrame(predicted_stock_price, index = range(df_train_orig.shape[0],df_train_orig.shape[0]+predicted_stock_price.size) )
    
    # Visualising the results
    plt.figure(figsize=(20, 10))

    plt.plot(df_train_orig, color = 'teal')
    plt.plot(df_test_orig, color = 'turquoise', label = 'Real [' + str(stock_name) + '] Stock Price' )

    plt.plot(predicted_stock_priceDF, color = 'brown', label = 'Predicted [' + str(stock_name) + '] Stock Price' )

    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()    

  def predictMulti(self, train_percentage, epoch, stock_name = None, stock_index = None, save_model = None):
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
    scaler = MinMaxScaler( feature_range = (0, 1) )
    stock_values_scaled_transposed = scaler.fit_transform(self.stock_values.T)
    stock_values_scaled_transposedDF = pd.DataFrame(stock_values_scaled_transposed)

    linscaler = MinMaxScaler( feature_range = (0, 1) )
    linscaler.fit_transform(self.stock_values.iloc[[stock_index]].T)

    #rescale
    rescaled_values = scaler.inverse_transform(stock_values_scaled_transposedDF)
    rescaled_valuesDF = pd.DataFrame(rescaled_values)

    #spliting train and test data
    df_train, df_test = our_train_test_split(stock_values_scaled_transposedDF, train_percentage)
    print('train size', df_train.shape, 'test size', df_test.shape)
    df_train_orig, df_test_orig = our_train_test_split(rescaled_valuesDF, train_percentage)
    
    #extracting X,y train
    X_train, y_train = getXy( df_train.to_numpy(), 25, df_train.shape[0] )
    print('x train size', X_train[stock_index].shape, 'y train size', y_train[stock_index].shape)

    #extracting X train
    X_test, y_test = getXy( df_test.to_numpy(), 25, df_test.shape[0] )

    if self.multimodel == None:
      self.multimodel = LSTMtrain(X_train, y_train, epoch)
    
    if save_model != None:
      self.multimodel.save(save_model)
    
    predicted_stock_price = self.multimodel.predict(X_test[stock_index])
    print("pred: ", predicted_stock_price.shape)
  
    #scaling
    predicted_stock_price = linscaler.inverse_transform(predicted_stock_price)
    predicted_stock_priceDF = pd.DataFrame(predicted_stock_price, index = range(df_train_orig.shape[0],df_train_orig.shape[0]+predicted_stock_price.size) )
    # print(predicted_stock_priceDF)
    # Visualising the results
    plt.figure(figsize=(20, 10))

    plt.plot(df_train_orig[stock_index], color = 'teal')
    plt.plot(df_test_orig[stock_index], color = 'turquoise', label = 'Real [' + str(stock_name) + '] Stock Price' )

    plt.plot(predicted_stock_priceDF, color = 'brown', label = 'Predicted [' + str(stock_name) + '] Stock Price' )

    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    print(predicted_stock_price.shape)
    print(predicted_stock_price.size)


#reding command line arguments
i = 1
dataset_path = None
number_of_timeseries = None
saved_multi_model_arg = None
while i <len(sys.argv):
  if sys.argv[i].strip().lower() == '-d':
    i += 1
    dataset_path = sys.argv[i]
  elif sys.argv[i].strip().lower() == '-n':
    i += 1
    number_of_timeseries = int(sys.argv[i])
  elif sys.argv[i].strip().lower() == '--pretrained':
    i += 1
    saved_multi_model_arg = sys.argv[i]
  i += 1

if dataset_path == None or number_of_timeseries == None:
  print('forecast.py: not enough arguments, check usage!')
  exit(-1)

if saved_multi_model_arg != None:
  if isdir(saved_multi_model_arg) == False:
    print('forecast.py: ' + saved_multi_model_arg + ' there is no such folder, pre trained model will not be used!')

sp = StockPredictor(dataset_path, saved_multi_model=saved_multi_model_arg)

for i in range(0,number_of_timeseries):
  sp.predictUni(0.3,5,stock_index = i, save_model = None, saved_model=None)

  sp.predictMulti(0.3,1,stock_index = i, save_model = None)
