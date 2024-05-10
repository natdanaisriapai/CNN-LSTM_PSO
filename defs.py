import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pytz
import json
import requests
import pandas as pd
import numpy as np
import pickle
import random
import warnings
from pytictoc import TicToc
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil import parser
from tqdm import tqdm_notebook
import ta as ta
from ta import add_all_ta_features
from ta.utils import dropna
import scipy
from scipy.signal import argrelmin, argrelmax
from scipy.signal import savgol_filter
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Input, Conv1D, MaxPooling1D, Flatten, GRU, Dropout, Bidirectional, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from keras import backend as K
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, MinMaxScaler, QuantileTransformer, OneHotEncoder
import keras_tuner as kt
from keras.layers import Bidirectional
import itertools
from itertools import permutations
# from keras.utils.vis_utils import plot_model
from keras.utils import plot_model
from dateutil.relativedelta import relativedelta

# Configure TensorFlow session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.config.experimental.enable_op_determinism()
tf.compat.v1.keras.backend.set_session(sess)

# Set seed for all random functions
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
tf.keras.utils.set_random_seed(0)

# Suppress warnings
warnings.filterwarnings("ignore")
# Configure TensorFlow session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.config.experimental.enable_op_determinism()
tf.compat.v1.keras.backend.set_session(sess)

# Set seed for all random functions
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
tf.keras.utils.set_random_seed(0)

# Suppress warnings
warnings.filterwarnings("ignore")
#############################################
#  SETTING
#############################################
symbols = 'CL=F'
# Available kline size "1m", "5m", "15m", "1h", "4h", "1d" for binance
kline_size = '30m'
trd_fee_pct = 0.0 # trading fee in percent
trd_fee_frac = trd_fee_pct/100
hld_fee_pct_8hr = 0 # holding fee in percent / 8hr (normal rate in Binance Future = 0.02%)
hld_fee_frac_hr = hld_fee_pct_8hr/(8*100)
OHLC_features = ['Open', 'High', 'Low', 'Close']
price_features = ['Close', 'Volume', 'Rel_Open', 'Rel_High', 'Rel_Low']
ind_features = ['Rel_SMA_fast', 'Rel_SMA_slow', 'MACD', 'ADX', 'CCI', 'ROC', 'RSI', 'TSI', '%K', '%D', '%R', 
                'Rel_BB_hband', 'Rel_BB_lband', 'Rel_ATR', 'UI',   'CMF', 'FI', 'MFI', 'VPT', 'VWAP']
#'ADI', 'OBV' ,'MFI', 'VPT', 'VWAP'
# time_features = ['Hour_of_Day', 'Day_of_Week', 'Day_of_Month', 'Day_of_Year']
time_features = ['Day_of_Week', 'Day_of_Month', 'Day_of_Year']
trans_time_features = [f(x) for x in time_features for f in (lambda x: x + '_Sin',lambda x: x + '_Cos')]
features = price_features + trans_time_features + ind_features
targets_scale = ['label']
targets_not_scale = []
targets = targets_scale + targets_not_scale
targets2 = ['label_1',
            ] # usd only for evaluation ploting
rolling_indicator = False

# model_no: 1 = CNN_LSTM, 2 = CNN, 3 = LSTM, 4 = LSTM_CNN, 5 = Transformer
model_no = 4
test_year = 2018
case_no = 1
cnn_no_node_list = [64, 128]
lstm_no_node_list = [64, 128]

split_type = 'time' # 'fraction', 'time'

# for split_type = 'fraction'
val_fraction = 0.1
test_fraction = 0.1
train_fraction = 1 - val_fraction - test_fraction
# for split_type = 'time'

T = 32 # T-days window of input data for predicting target_class
n_ind = 10 # indicator fast
k_ind = 15 # indicator slow
start_index_delta = - T - k_ind - 23 # import more bar befor time_beg for labeling
end_index_delta = 2 # import more bar after time_end for labeling

# Pre-procession features
pct_change_transform_features = ['Close', 'Volume', ] # non-stationary feasture that need to transform to stationary (transform be percent change from previous time-step)
quantile_transform_features = ['Volume', 'VPT'] # Transform feature with obviuos outliner
# quantile_transform_features = ['Volume']


train_model_ = True
features_plotting = False
save_model_ = True
load_model_ = not train_model_
def df_extract_feature(df, rolling_indicator = False):
  df['Rel_Open'] = (df['Open'] - df['Close'])/df['Close']
  df['Rel_High'] = (df['High'] - df['Close'])/df['Close']
  df['Rel_Low'] = (df['Low'] - df['Close'])/df['Close']

  # Create Indicator Column

  # SMA
  df['Rel_SMA_fast'] = (ta.trend.SMAIndicator(close=df['Close'], window=n_ind, fillna=False).sma_indicator() - df['Close']) / df['Close'] 
  df['Rel_SMA_slow'] = (ta.trend.SMAIndicator(close=df['Close'], window=k_ind, fillna=False).sma_indicator() - df['Close']) / df['Close'] 

  # CCI
  df['CCI'] = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=n_ind, constant=0.015, fillna=False).cci()

  # ROC
  df['ROC'] = ta.momentum.ROCIndicator(close=df['Close'], window=n_ind, fillna=False).roc()

  # %R
  df['%R'] = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=n_ind, fillna=False).williams_r()

  # MFI
  df['MFI'] = ta.volume.MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=n_ind, fillna=False).money_flow_index()

  # VPT
  df['VPT'] = ta.volume.VolumePriceTrendIndicator(close=df['Close'], volume=df['Volume'], fillna=False).volume_price_trend()
  df.loc[df.index[0], 'VPT'] = np.nan
  df.loc[df.index[1], 'VPT'] = np.nan = np.nan # correct library bug. the first 2 row in VPT should be np.nan

  # VWAP
  df['VWAP'] = (ta.volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=n_ind, fillna=False).volume_weighted_average_price()- df['Close']) / df['Close']

  # MACD
  df['MACD'] = ta.trend.macd(close=df['Close'], window_slow=k_ind, window_fast=n_ind, fillna=False)

  df['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=n_ind, fillna=False).adx()

  # RSI
  df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=n_ind, fillna=False).rsi()

  # TSI
  df['TSI'] = ta.momentum.TSIIndicator(close=df['Close'], window_slow=k_ind, window_fast=n_ind, fillna=False).tsi()

  # %K
  df['%K'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=n_ind, smooth1=n_ind, smooth2=n_ind, fillna=False).stochrsi_k()

  # %D
  df['%D'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=n_ind, smooth1=n_ind, smooth2=n_ind, fillna=False).stochrsi_d()

  # Bollinger Bands
  df['Rel_BB_hband'] = (ta.volatility.BollingerBands(close=df['Close'], window=n_ind, window_dev=2, fillna=False).bollinger_hband() - df['Close']) / df['Close']
  df['Rel_BB_lband'] = (ta.volatility.BollingerBands(close=df['Close'], window=n_ind, window_dev=2, fillna=False).bollinger_lband() - df['Close']) / df['Close']

  # ATR
  df['Rel_ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=n_ind, fillna=False).average_true_range() / df['Close']

  # UI
  df['UI'] = ta.volatility.UlcerIndex(close=df['Close'], window=n_ind, fillna=False).ulcer_index()

  # CMF
  df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=n_ind, fillna=False).chaikin_money_flow()

  # FI 
  df['FI'] = ta.volume.ForceIndexIndicator(close=df['Close'], volume=df['Volume'], window=n_ind, fillna=False).force_index()

  return df

def Xy4nn(df, feature_columns, target_columns, T):
    X, y = [], []
    date_list = []

    for i in range(df[target_columns].shape[0] - (T - 1)):
        # Append the date to the date_list
        date_list.append(df[['Time']].iloc[i + (T - 1)].values)

        # Append the input features and target labels to X and y
        X.append(df[feature_columns].iloc[i:i+T].values)
        y.append(df[target_columns].iloc[i + (T - 1)].values)

    # Convert X and y to NumPy arrays
    X, y = np.array(X), np.array(y).reshape(-1, len(target_columns))

    return X, y, date_list
def df_identifying_label(df, span):
    # Calculate smoothed price using Savitzky-Golay filter
    df['Smoothed_Price*'] = smooth_data_savgol_2(np.array(df['Close']), span)

    # Calculate smoothed return as the percent change of smoothed price
    df['Smoothed_Return*'] = df['Smoothed_Price*'].pct_change(1).shift(periods=-1)

    return df
def smooth_data_savgol_2(arr, span):  
    return savgol_filter(arr, span * 2 + 1, 1)
def train_stock_market_model(df, name):

    # Preprocess the data
    df['Time'] = pd.to_datetime(df['Time']).dt.strftime("%Y-%m-%d 00:00:00")
    df['label_1'] = df['Close'].pct_change(1).shift(periods=-1).copy()
    df = df_extract_feature(df).copy()
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    scaler = MinMaxScaler()
    scaler_label = MinMaxScaler()

    # Define the feature and target columns
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                       'Rel_Open', 'Rel_High', 'Rel_Low', 'Rel_SMA_fast', 'Rel_SMA_slow',
                       'CCI', 'ROC', '%R', 'MFI', 'VPT', 'VWAP', 'MACD', 'ADX', 'RSI', 'TSI',
                       '%K', '%D', 'Rel_BB_hband', 'Rel_BB_lband', 'Rel_ATR', 'UI', 'CMF',
                       'FI']

    target_columns = ['label_1']

    original_df = df.copy()

    df['date'] = pd.to_datetime(df['Time']).dt.date

    train_data = df.iloc[:int(len(df)*0.8),:]
    val_data = df.iloc[int(len(df)*0.8):int(len(df)*0.8+len(df)*0.1),:]
    test_data = df.iloc[int(len(df)*0.8+len(df)*0.1):,:]
    # Scale the data
    train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    train_data[target_columns] = scaler_label.fit_transform(train_data[target_columns])

    val_data[feature_columns] = scaler.transform(val_data[feature_columns])
    val_data[target_columns] = scaler_label.transform(val_data[target_columns])

    test_data[feature_columns] = scaler.transform(test_data[feature_columns])
    test_data[target_columns] = scaler_label.transform(test_data[target_columns])

    # Prepare the input data for the model
    X_train, y_train, train_date = Xy4nn(train_data, feature_columns, target_columns, T)
    X_val, y_val, val_date = Xy4nn(val_data, feature_columns, target_columns, T)
    X_test, y_test, test_date = Xy4nn(test_data, feature_columns, target_columns, T)

    train_date = np.array(train_date).reshape(-1)
    val_date = np.array(val_date).reshape(-1)
    test_data_date = np.array(test_date).reshape(-1)

    N = X_train.shape[2]

    BATCH = 128
    EPOCH = 100
    LR = 1e-7

    lstm_no_node_list = [32, 64, 128]
    cnn_no_node_list = [32, 64, 128]
    kernel_size_list = [7]
    best_mse = 9999999

    object_dict = {
        'feature_columns': feature_columns,
        'scaler': scaler,
        'scaler_label': scaler_label,
        'target_columns': target_columns
    }
    print(X_train.shape)
    # Iterate over combinations of hyperparameters
    for cnn_no_node, lstm_no_node, kernel_size in list(list(itertools.product(cnn_no_node_list, lstm_no_node_list, kernel_size_list))):
        input1 = Input(shape=(T, N))
        # Define the model architecture
        x = Conv1D(filters=cnn_no_node, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(input1)
        x = LSTM(lstm_no_node, return_sequences=True)(x)
        x = Dropout(0.6)(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(64)(x)
        x = Dropout(0.4)(x)
        out1 = Dense(1, activation='sigmoid', name='class1')(x)

        model = Model(inputs=input1, outputs=[out1])

        model.compile(optimizer=Adam(learning_rate=LR),
                      loss='MeanSquaredError',
                      metrics='MeanSquaredError')

        try:
            model.load_weights(str(name)+' model.h5')
        except:
            pass
        model.save_weights(str(name)+' model.h5')

        K.set_value(model.optimizer.lr, LR)

        d_k = 256
        d_v = 256
        n_heads = 12
        ff_dim = 256
        lr_decay = ReduceLROnPlateau(monitor='val_loss',
                                     patience=10, verbose=0,
                                     factor=0.95, min_lr=1e-3, cooldown=0)

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0,
                                   patience=30, verbose=0, mode='auto',
                                   baseline=None, restore_best_weights=True)

        # Train the model
        History = model.fit(X_train, y_train,
                            epochs=EPOCH,
                            batch_size=BATCH,
                            validation_split=0.0,
                            validation_data=(X_val, y_val),
                            shuffle=True, verbose=0,
                            callbacks=[lr_decay, early_stop])

        # Evaluate model performance
        result_train = get_model_performance_result(model, X_train, y_train)
        result_val = get_model_performance_result(model, X_val, y_val)
        result_test = get_model_performance_result(model, X_test, y_test)

        # Update the best model and MSE if the current model performs better on the validation set
        if best_mse > float(result_val['MSE'][0]):
            best_model = model
            best_mse = float(result_val['MSE'][0])
            best_model = model
            # print('better')
        # print(cnn_no_node, lstm_no_node, kernel_size,result_val)

    # Save the best model and associated objects
    best_model.save(str(name)+' model.h5')
    with open('objects.pkl', 'wb') as f:
        pickle.dump(object_dict, f)

def get_model_performance_result(model, X, y):
    """
    Computes the performance metrics of a trained model on the given input data.

    Args:
        model (Model): Trained machine learning model.
        X (ndarray): Input features.
        y (ndarray): Target values.

    Returns:
        dict: Dictionary containing the computed loss and mean squared error (MSE).

    """

    # Predict the target values
    y_pred = model.predict(X, verbose=0)

    # Compute the loss and mean squared error (MSE)
    loss, mse = model.evaluate(X, y, verbose=0)

    # Prepare the result data
    data = {
        'LOSS': [loss],
        'MSE': [mse]
    }

    return data

##########polygon##########
from polygon import RESTClient
def polygon(ticker, multiplier, timespan, start, end):
    client = RESTClient(api_key='7PZTsP8NHCZy0cEsWRlgFaSa2Q8zm8Zb')
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=multiplier, timespan=timespan, from_=start, to=end, limit=50000):
        aggs.append(a)
    df = pd.DataFrame(aggs)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.timestamp, unit='ms', utc=True)).tz_convert('US/Eastern')
    df.drop(['timestamp', 'transactions', 'otc'], axis=1, inplace=True)
    return df

# def polygon(stock_name,tf_min,date_start1,date_start2,date_end1,date_end2):
#     date_start = pd.date_range(date_start1,date_start2 , freq='1M')-pd.offsets.MonthBegin(1)
#     date_end = pd.date_range(date_end1,date_end2 , freq='1M')-pd.offsets.MonthEnd(1)
#     start = []
#     for i in date_start:
#      ss = datetime.date(i)
#      start.append(ss)
#     end = []
#     for i in date_end:
#      dd = datetime.date(i)
#      end.append(dd)
#     url = []
#     for s,e in zip(start,end):
#         h = 'https://api.polygon.io/v2/aggs/ticker/'+stock_name+'/range/'+tf_min+'/minute/'+str(s)+'/'+str(e)+'?adjusted=true&sort=asc&limit=50000&apiKey=XrpnQBzsuL5AKxabLuN0jzgDP8Uk3IL1'
#         # print(h)
#         url.append(h)
#     data = pd.DataFrame()
#     for u,s in zip(url,start):
#         r = requests.get(f'{u}')
#         df = pd.DataFrame(r.json()['results'])
#         # print(len(df))
#         est = pytz.timezone('US/Eastern')
#         utc = pytz.utc
#         df.index = [datetime.utcfromtimestamp(ts / 1000.).replace(tzinfo=utc).astimezone(est) for ts in df['t']]
#         df.index.name = 'Time'
#         df.columns = ['Volume', 'Volume Weighted', 'Open', 'Close', 'High', 'Low', 'Time Stamp', 'Num Items']
#         data = pd.concat([data, pd.DataFrame(df)])
#     return(data)
def predict_stock_price(data, PATH_MODEL):
    """
    Predicts the stock price based on the given data using a trained model.

    Args:
        data (DataFrame): Input data for prediction.
        PATH_MODEL (str): Path to the saved model.

    Returns:
        list: List of dictionaries representing the predicted stock prices.

    """

    # Load the trained model
    model = load_model(PATH_MODEL + '/model.h5')

    # Load the saved objects
    with open(PATH_MODEL+'/objects.pkl', 'rb') as f:
        object_dict = pickle.load(f)

    feature_columns = object_dict['feature_columns']
    scaler = object_dict['scaler']
    scaler_label = object_dict['scaler_label']
    target_columns = object_dict['target_columns']

    # Prepare the input data
    df = data.copy()
    df = df_extract_feature(df).copy()
    df = df.dropna()

    df = df[-T:]  # Assuming T is defined somewhere
    df[feature_columns] = scaler.transform(df[feature_columns])

    X = []
    for i in range(df.shape[0] - (T-1)):
        X.append(df[feature_columns].iloc[i:i+T].values)
    X = np.array(X)

    BATCH = 128
    y_pred = model.predict(X, batch_size=BATCH, verbose=0)

    # Prepare the output data
    last_price_df = data[['Stock', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(1).copy()
    last_price_df['y_pred'] = y_pred
    last_price_df['y_pred'] = scaler_label.inverse_transform(np.array(last_price_df['y_pred']).reshape(-1, 1))
    last_price_df['Close_Predict'] = last_price_df['Close'] + (last_price_df['Close'] * last_price_df['y_pred'])
    last_price_df['High_Predict'] = last_price_df['Close_Predict'] + (last_price_df['Close_Predict'] * 0.1)
    last_price_df['Low_Predict'] = last_price_df['Close_Predict'] - (last_price_df['Close_Predict'] * 0.1)

    return last_price_df.to_dict('records')


# from polygon import WebSocketClient
# from polygon.websocket.models import WebSocketMessage
# from typing import List


# price = pd.DataFrame(columns=['Open', 'Close', 'High', 'Low'])
# est = pytz.timezone('US/Eastern')
# utc = pytz.utc

# def handle_msg(msg: List[WebSocketMessage]):
#     for m in msg:
#         print(m)
#         unix_time = m.start_timestamp
#         human_time = datetime.utcfromtimestamp(unix_time / 1000).replace(tzinfo=utc, microsecond=0).astimezone(est)
#         Open = m.open
#         Close = m.close
#         High = m.high
#         Low = m.low
#         End_Time = datetime.utcfromtimestamp(m.end_timestamp / 1000).replace(tzinfo=utc, microsecond=0).astimezone(est)
#         Volume =m.volume

#         price.loc[human_time] = [Open, Close, High, Low,human_time,End_Time, Volume]
#     return(price)