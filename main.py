import math
from datetime import date
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plot
plot.style.use('ggplot')


def closing_prices(company):
    print("The program will use 80% of the data to train and the rest will be used for testing.")
    print("Please enter a starting date (ex. 2012-01-01 [yyyy-MM-DD]): ")
    start = input()
    print("Please enter a closing date (blank for today): ")
    end = input()
    if(len(end) == 0):
      end = date.today()
    df = web.DataReader(company, data_source='yahoo', start=start, end=end)
    #Visualize the closing price history
    plot.figure(figsize=(16,8))
    plot.title('Close Price History - {}'.format(company))
    plot.plot(df['Close'])
    plot.xlabel('Date',fontsize=18)
    plot.ylabel('Close Price USD ($)',fontsize=18)
    plot.show()

def test_ml(company):
    print("The program will use 80% of the data to train and the rest will be used for testing.")
    print("Please enter a starting date (ex. 2012-01-01 [yyyy-MM-DD]): ")
    start = input()
    print("Please enter a closing date (blank for today): ")
    end = input()
    if(len(end) == 0):
      end = date.today()
    df = web.DataReader(company, data_source='yahoo', start=start, end=end)
    data = df.filter(['Close'])
    dataset = data.values
    #Compute the number of rows to train the model on
    training_data_len = math.ceil( len(dataset) *.8) 
    #Preprocessing the data, scale to 0 thru 1
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)

    #Create the scaled training data set 
    train_data = scaled_data[0:training_data_len  , : ]
    x_train=[]
    y_train = []
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0]) # Store adjusted stock prices for the last 60 days (not including curr day)
        y_train.append(train_data[i,0])     # Store adjusted stock price for current day

    print("Getting training data...")
    #Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #Reshape the data into the shape accepted by the LSTM (3-dimensional)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    print("Creating model...")
    #Build the LSTM network model with two 50 neurons layers and two dense layers, one with 25 and the other with 1 neuron.
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    print("Compiling model...")
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Training model... (This part may take a while)")
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    print("Creating test sets...")
    test_data = scaled_data[training_data_len - 60: , : ]
    #Create the x_test and y_test data sets
    x_test = []
    y_test =  dataset[training_data_len : , : ] #Get rest of data
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])

    #Convert x_test to a numpy array 
    x_test = np.array(x_test)

    #Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    print("Getting predicted values...")
    #Getting the models predicted price values
    predictions = model.predict(x_test) 
    predictions = scaler.inverse_transform(predictions)#Undo scaling

    print("Done!")
    
    #Calculate/Get the value of RMSE
    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
    print("RSME = {}".format(rmse))


    #Plot/Create the data for the graph
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    #Visualize the data
    plot.figure(figsize=(16,8))
    plot.title('Close Price Model - {}'.format(company))
    plot.xlabel('Date', fontsize=18)
    plot.ylabel('Close Price USD ($)', fontsize=18)
    plot.plot(train['Close'])
    plot.plot(valid[['Close', 'Predictions']])
    plot.legend(['Training Data', 'Real Data', 'Predictions'], loc='lower right')
    plot.show()

def predict(company):
    print("The program will attempt to predict tomorrow's closing price.")
    print("Please enter a starting date for training (ex. 2012-01-01 [yyyy-MM-DD]): ")
    start = input()
    end = date.today()
    df = web.DataReader(company, data_source='yahoo', start=start, end=end)
    data = df.filter(['Close'])
    dataset = data.values
    #Model will train on entire dataset
    training_data_len = len(dataset)
    #Preprocessing the data, scale to 0 thru 1
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)

    #Create the scaled training data set 
    train_data = scaled_data[0:training_data_len  , : ]
    x_train=[]
    y_train = []
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0]) # Store adjusted stock prices for the last 60 days (not including curr day)
        y_train.append(train_data[i,0])     # Store adjusted stock price for current day

    print("Getting training data...")
    #Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #Reshape the data into the shape accepted by the LSTM (3-dimensional)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    print("Creating model...")
    #Build the LSTM network model with two 50 neurons layers and two dense layers, one with 25 and the other with 1 neuron.
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    print("Compiling model...")
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Training model... (This part may take a while)")
    model.fit(x_train, y_train, batch_size=1, epochs=1)
   
    print("Creating test set...")
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print("Getting predicted values...")
    pred_price = model.predict(X_test)

    print("Done!")
    pred_price = scaler.inverse_transform(pred_price)
    print("Tomorrow's predicted close price for {comp} is {price}".format(comp=company, price=pred_price[0][0]))

def main():
    print("Welcome to Stock Price Predictor! This program will use ML to predict stock prices")
    print("Please enter the trading symbol you would like to examine (ex TSLA or ^DJI): ")
    company = input()
    try:
        df = web.DataReader(company, data_source='yahoo', start="2012-01-01", end="2012-01-05")
    except:
        print("Error trading symbol not valid")
        sys.exit()
    print("What would you like to do? *Note: all starting days must be at least 60 days before the end date")
    print("  1. View Closing Prices (History)")
    print("  2. Test the ML Model")
    print("  3. Predict Stock Price")
    choice = input("Option: ")
    if(choice == "1"):
        closing_prices(company)
    elif(choice == "2"):
        test_ml(company)
    elif(choice == "3"):
        predict(company)
    
    else:
        print("Guess you haven't made up your mind. Please try again later.")


if __name__ == "__main__":
    main()