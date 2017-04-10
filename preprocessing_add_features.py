import numpy as np
import os
import time
import pandas as pd


holidays = {
    'moon': {
        'begin': time.strptime('2016-09-15', '%Y-%m-%d'),
        'end': time.strptime('2016-09-17 23:59:59', '%Y-%m-%d %H:%M:%S')
    },
    'national': {
        'begin': time.strptime('2016-10-01', '%Y-%m-%d'),
        'end': time.strptime('2016-10-07 23:59:59', '%Y-%m-%d %H:%M:%S')
    }
}
routes_df = pd.read_csv('dataSets/training/routes (table 4).csv')
routes = dict()
for i in range(routes_df.shape[0]):
    key = str(routes_df.loc[i, 'intersection_id']) + ',' + str(routes_df.loc[i, 'tollgate_id'])
    val = str(routes_df.loc[i, 'link_seq']).split(',')
    routes[key] = val

links_df = pd.read_csv('dataSets/training/links (table 3).csv')
links = dict()
for i in range(links_df.shape[0]):
    key = str(links_df.loc[i, 'link_id'])
    val = [float(links_df.loc[i, 'length']), float(links_df.loc[i, 'width']), float(links_df.loc[i, 'lanes'])]
    links[key] = val

travel_features_ndim = 9
volume_features_ndim = 10


def get_weather(in_filename):
    weather_file_postfix = os.path.basename(in_filename).split('_')[0]

    weather_filename = os.path.join(
        os.path.abspath(os.path.join(in_filename, os.pardir)),
        'weather (table 7)_' + weather_file_postfix + '.csv')
    weather_df = pd.read_csv(weather_filename)
    weather = dict()
    for i in range(weather_df.shape[0]):
        key = str(weather_df.loc[i, 'date']) + ',' + str(weather_df.loc[i, 'hour'])
        val = [float(weather_df.loc[i, 'wind_direction']), float(weather_df.loc[i, 'wind_speed']),
               float(weather_df.loc[i, 'temperature']), float(weather_df.loc[i, 'precipitation'])]
        weather[key] = val
    return weather


def build_travel_time_examples(in_filename):
    file_suffix = '_travel_time_dataset.csv'
    out_prefix = in_filename.split('_')[0]
    out_filename = out_prefix + file_suffix
    weather = get_weather(in_filename)
    travel_df = pd.read_csv(in_filename, dtype=str)
    print('dataset size: ', travel_df.shape)
    print('cleaning...')
    travel_ds = np.ndarray(shape=[travel_df.shape[0], travel_features_ndim+1], dtype=np.float)
    # features: <route_quality,
    #           wind_direction, wind_speed, temperature, precipitation,
    #           weekend, time_of_day>
    for i in range(0, travel_df.shape[0]):
        # calculate the route's quality.
        # quality = length / (width * lanes)
        link_seq = routes.get(travel_df.loc[i, 'intersection_id'] + ',' + travel_df.loc[i, 'tollgate_id'])
        links_quality = []
        for link_id in link_seq:
            links_quality.append(links.get(link_id)[0] / (links.get(link_id)[1] * links.get(link_id)[2]))
        travel_ds[i, 0] = float(str.format('%.2f' % np.sum(links_quality)))
        time_window = travel_df.loc[i, 'time_window'][1: -1].split(',')
        start_time = time.strptime(time_window[0], '%Y-%m-%d %H:%M:%S')
        travel_ds[i, 1:5] = weather.get(time_window[0].split(' ')[0] + ',' + str(start_time.tm_hour//3 * 3))
        travel_ds[i, 5] = 1 if start_time.tm_wday in [5, 6] else 0
        travel_ds[i, 6] = start_time.tm_hour * 3 + start_time.tm_min // 20
        travel_ds[i, 7] = 1 if (start_time >= holidays['moon']['begin'] and start_time < holidays['moon']['end'])\
            or (start_time >= holidays['national']['begin'] and start_time < holidays['national']['end']) else 0
        travel_ds[i, 8] = 1 if start_time >= holidays['national']['begin'] and start_time < holidays['national']['end'] else 0
        travel_ds[i, 9] = float(travel_df.loc[i, 'avg_travel_time'])

    dataset = pd.DataFrame(data=travel_ds, dtype=float, columns=['route_quality', 'wind_direction',
                                                                 'wind_speed', 'temperature',
                                                                 'precipitation', 'weekend',
                                                                 'time_of_day', 'holiday',
                                                                 'free', 'avg_travel_time'
                                                                 ])
    dataset.to_csv(path_or_buf=out_filename)
    print('finished.')

build_travel_time_examples('dataSets/testing-phase1/test1_20min_avg_travel_time.csv')
build_travel_time_examples('dataSets/training/training_20min_avg_travel_time.csv')

def build_volume_examples(in_filename):
    file_suffix = '_volume_dataset.csv'
    out_prefix = in_filename.split('_')[0]
    out_filename = out_prefix + file_suffix
    weather = get_weather(in_filename)
    volume_df = pd.read_csv(in_filename, dtype=str)
    print('dataset size: ', volume_df.shape)
    print('cleaning...')
    volume_ds = np.ndarray(shape=[volume_df.shape[0], volume_features_ndim+1], dtype=np.float)
    # features: <tollgate_scale,
    #           wind_direction, wind_speed, temperature, precipitation,
    #           weekend, time_of_day,
    #           direction>
    tollgate_scale = dict()
    for inter_toll, link_seq in routes.items():
        links_quality = []  # the shorter, the better
        for link_id in link_seq:
            links_quality.append(links.get(link_id)[0] / (links.get(link_id)[1] * links.get(link_id)[2]))
        tollgate_id = inter_toll.split(',')[1]
        if tollgate_id in tollgate_scale.keys():
            tollgate_scale[tollgate_id] += np.divide(1.0, np.sum(links_quality))
        else:
            tollgate_scale[tollgate_id] = np.divide(1.0, np.sum(links_quality))
    for i in range(volume_df.shape[0]):
        # calculate the scale of tollgate
        volume_ds[i, 0] = float(str.format('%.3f' % tollgate_scale.get(volume_df.loc[i, 'tollgate_id'])))
        time_window = volume_df.loc[i, 'time_window'][1: -1].split(',')
        start_time = time.strptime(time_window[0], '%Y-%m-%d %H:%M:%S')
        volume_ds[i, 1:5] = weather.get(time_window[0].split(' ')[0] + ',' + str(start_time.tm_hour // 3 * 3))
        volume_ds[i, 5] = 1 if start_time.tm_wday in [5, 6] else 0
        volume_ds[i, 6] = start_time.tm_hour * 3 + start_time.tm_min // 20
        volume_ds[i, 7] = float(volume_df.loc[i, 'direction'])
        volume_ds[i, 8] = 1 if (start_time >= holidays['moon']['begin'] and start_time < holidays['moon']['end'])\
            or (start_time >= holidays['national']['begin'] and start_time < holidays['national']['end']) else 0
        volume_ds[i, 9] = 1 if start_time >= holidays['national']['begin'] and start_time < holidays['national']['end'] else 0

        volume_ds[i, 10] = float(volume_df.loc[i, 'volume'])

    dataset = pd.DataFrame(data=volume_ds, dtype=float, columns=['tollgate_scale', 'wind_direction',
                                                                 'wind_speed', 'temperature',
                                                                 'precipitation', 'weekend',
                                                                 'time_of_day', 'direction',
                                                                 'holiday', 'free',
                                                                 'volume'
                                                                 ])
    dataset.to_csv(path_or_buf=out_filename)
    print('finished.')


build_volume_examples('dataSets/testing-phase1/test1_20min_avg_volume.csv')
build_volume_examples('dataSets/training/training_20min_avg_volume.csv')

build_travel_time_examples('dataSets/testing-phase1/submission_sample_travelTime.csv')
build_volume_examples('dataSets/testing-phase1/submission_sample_volume.csv')
