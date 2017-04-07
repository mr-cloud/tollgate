import numpy as np
import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn, layers, metrics


def travel_normalize(df):
    avg_std_container = np.ndarray(shape=[2, df.shape[1]], dtype=float)
    avg_std_container[0, :] = np.mean(df, axis=0)
    avg_std_container[1, :] = np.std(df, axis=0)
    return ((df - [[avg_std_container[0, 0], 0, avg_std_container[0, 2], avg_std_container[0, 3],
                   avg_std_container[0, 4], 0, avg_std_container[0, 6], 0]])
            / [[avg_std_container[1, 0], 1, avg_std_container[1, 2], avg_std_container[1, 3],
                avg_std_container[1, 4], 1, avg_std_container[1, 6], 1]]), avg_std_container
# csv dataset --> feature column --> TF Learn --> evaluation metrics
training_travel_df = pd.read_csv('dataSets/training/training_travel_time_dataset.csv', index_col=0)
test_travel_df = pd.read_csv('dataSets/testing-phase1/test1_travel_time_dataset.csv', index_col=0)
training_travel_df, _ = travel_normalize(training_travel_df)
print('training dataset size: ', training_travel_df.shape)
test_travel_df, _ = travel_normalize(test_travel_df)
print('test dataset size:', test_travel_df.shape)


num_call = 0


def travel_input_fn(df):
    global num_call
    num_call += 1
    print('input function was called %d times' % num_call)
    return {
               'route_quality': tf.constant(df.iloc[:, 'route_quality']),
               'wind_direction': tf.constant(df.iloc[:, 'wind_direction']),
               'wind_speed': tf.constant(df.iloc[:, 'wind_speed']),
               'temperature': tf.constant(df.iloc[:, 'temperature']),
               'precipitation': tf.constant(df.iloc[:, 'precipitation']),
               'weekend': tf.constant(df.iloc[:, 'weekend']),
               'time_of_day': tf.constant(df.iloc[:, 'time_of_day'])
           }, tf.constant(df.iloc[:, 'avg_travel_time'])

route_quality = layers.real_valued_column('route_quality')
wind_direction = layers.real_valued_column('wind_direction')
wind_direction_range = layers.bucketized_column(wind_direction, boundaries=[0, 45, 90, 135,
                                                                            180, 225, 270,
                                                                            315, 360])
wind_speed = layers.real_valued_column('wind_speed')
temperature = layers.real_valued_column('temperature')
precipitation = layers.real_valued_column('precipitation')
weekend = layers.real_valued_column('weekend')
time_of_day = layers.real_valued_column('time_of_day')
regressor = learn.LinearRegressor(feature_columns=[route_quality, wind_direction_range,
                                                   wind_speed, temperature,
                                                   precipitation, weekend,
                                                   time_of_day])


def travel_input_fn_training():
    return travel_input_fn(training_travel_df)


def travel_input_fn_test():
    return travel_input_fn(test_travel_df)


for i in range(10):
    regressor.fit(input_fn=travel_input_fn, steps=10)
    eval_rst = regressor.evaluate(input_fn=travel_input_fn_training,
                                  metrics={
                                      'my_mae': metrics.streaming_mean_absolute_error
                                  })
    print('Trainig MAPE: ', eval_rst['my_mae'])
test_rst = regressor.evaluate(input_fn=travel_input_fn_test,
                   metrics={
                       'my_mae': metrics.streaming_mean_absolute_error
                   })
print('Test MAPE:', test_rst['my_mae'])