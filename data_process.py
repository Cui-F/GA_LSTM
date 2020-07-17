import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

import os

def csv_to_dataset(file_path):

    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    if os.path.exists(file_path):
        data = pd.read_csv(file_path)

    data_ = data.values

    holiday = data_[:, 0]
    holiday_dict = dict(enumerate(set(holiday)))
    holiday_dict_ = {v:k for k,v in holiday_dict.items()}
    enc_holiday = [holiday_dict_[k] for k in holiday]

    weather_description = data_[:6]
    weather_dict = dict(enumerate(set(weather_description)))
    holiday_dict_ = {v: k for k, v in weather_dict.items()}
    enc_weather = [holiday_dict_[k] for k in weather_description]

    cloud_all = data_[:, 4]
    cloud_all = np.reshape(cloud_all, (-1, 1))
    nor_cloud = normalize(cloud_all, norm='max')
    nor_cloud_ = np.reshape(nor_cloud, (-1))
    





print(os.path.abspath(r'Metro_Interstate_Traffic_Volum.csv'))



data = pd.read_csv(r'Metro_Interstate_Traffic_Volume.csv')

all_data = data.values

holiday = all_data[:, 0]

temp = all_data[:, 1]

cloud_all = all_data[:, 4]
cloud_all = cloud_all.astype(np.int)

cloud_all = all_data[:, 4]
cloud_all = np.reshape(cloud_all, (-1, 1))
cloud_all_ = normalize(cloud_all, norm='max', axis=0)
cloud_all_ = np.reshape(cloud_all_, (-1))

test = set(holiday)



test1 = dict(enumerate(test))

test2 = {v:k for k,v in test1.items()}
#

enc_holiday = [test2[k] for k in holiday]
# holiday = csv[:][0]
#
# temp = csv[:][1]
#
# # min_max_scaler = preprocessing.minmax_scale()
# # temp = min_max_scaler(temp)
#
# test_data = data['traffic_volume']
# test_data.index = data['date_time']
#
# test_data.plot(subplots=True)
#
# uni_data = test_data.values
#
#
# print(data['holiday'])
#
# print(data['temp'])
#
data.head()