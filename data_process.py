import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, minmax_scale, MinMaxScaler
import os



def date_split(date):

    date_, time_ = date.split(' ')
    year_, month_, day_ = date_.split('-')
    hour_, _, _ = time_.split(':')
    return int(year_), int(month_), int(day_), int(hour_)


def csv_to_dataset(file_path):

    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    if os.path.exists(file_path):
        data = pd.read_csv(file_path)

    data_ = data.values

    holiday = data_[:, 0].astype(np.str).tolist()
    fes = list(set(holiday))
    fes.sort(key = holiday.index)
    holiday_dict = dict(enumerate(fes))
    holiday_dict_ = {v:k for k,v in holiday_dict.items()}
    enc_holiday = [holiday_dict_[k] for k in holiday]

    weather_description = data_[:, 6].astype(np.str).tolist()
    description = list(set(weather_description))
    description.sort(key=weather_description.index)
    weather_dict = dict(enumerate(description))
    holiday_dict_ = {v: k for k, v in weather_dict.items()}
    enc_weather = [holiday_dict_[k] for k in weather_description]

    cloud_all = data_[:, 4]
    cloud_all = np.reshape(cloud_all, (-1, 1))
    nor_cloud = normalize(cloud_all, norm='max')
    nor_cloud_ = np.reshape(nor_cloud, (-1))

    temp = data_[:, 1]
    temp = np.reshape(temp, (-1, 1))
    nor_temp = minmax_scale(temp, axis=0)
    nor_temp_ = np.reshape(nor_temp, (-1))

    date_time = data_[:, 7]
    date_time = date_time.astype(np.str)
    r = list(map(date_split, date_time))
    r_ = np.array(r, dtype=np.float)
    year_ = r_[:, 0]
    month_ = r_[:, 1]
    day_ = r_[:, 2]
    hour_ = r_[:, 3]

    label = data_[:, 8].astype(np.int)
    label = np.reshape(label, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(label)
    nor_label = scaler.transform(label)

    rain_1h = data_[:, 2]

    snow_1h = data_[:, 3]

    all_data_ = np.array(list(zip(enc_holiday, nor_temp_, rain_1h, snow_1h,enc_weather,  year_, month_, day_, hour_, nor_label.reshape((-1)))))
    return all_data_, nor_label
