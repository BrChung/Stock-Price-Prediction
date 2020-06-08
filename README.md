# Stock Price Prediction

This script uses Python and Machine Learning (LSTM) to predict the future stock closing prices using the last 60 days.

![demo](https://raw.githubusercontent.com/BrChung/Stock-Price-Prediction/master/assets/example.png)

## Getting Started

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

OR

Use Google Colab to execute the [code](https://colab.research.google.com/drive/18nU8S1pkfEyPpD94mVhuAl-QJw0SZNnE?usp=sharing).

## Usage

The program is created with Python3 in mind.

```bash
python3 main.py
```

The program comes with a basic command line interface where users can select between the three following options: 1. Viewing closing prices 2. Testing the ML model 3. Predicting tomorrow's closing price

The user may select the stock they wish to view by its trading symbol (ex. ^DJI for Dow Jones Industrial Average)
and the starting date for the machine learning model's training.

The program queries Yahoo Finance to retrieve the stock data. The program was inspired and referenced from randerson112358's [Medium article](https://medium.com/@randerson112358/stock-price-prediction-using-python-machine-learning-e82a039ac2bb).
