import pandas as pd
import numpy as np

np.random.seed(980525)


def mask(x, target_rate=0.2):
    shapes = x.shape
    x = x.reshape(-1)
    idx = np.random.permutation(x.shape[0])
    x[idx[:int(target_rate * x.shape[0])]] = np.nan
    return x.reshape(shapes)


def generate_dataset(dataset, miss_rate=0.2):
    name_dict = {'pm25': 'PM25/pm25_ground.txt', 'activity': 'activity.csv', 'traffic': 'traffic.csv', 'solar': 'solar_AL.csv'}
    dim_dict = {'pm25': (1, None), 'activity': (4, 7), 'traffic': (3, None), 'solar': (None, None)}
    data = pd.read_csv('dataset/' + name_dict[dataset], delimiter=',').iloc[:, dim_dict[dataset][0]:dim_dict[dataset][1]].values
    np.savetxt('dataset/' + dataset + '.txt', data, delimiter=',')
    rate_0 = np.isnan(data).sum() / data.shape[0] / data.shape[1]
    data = mask(data, miss_rate)
    np.savetxt('dataset/' + dataset + '_' + str(int(miss_rate * 10)) + '.txt', data, delimiter=',')
    rate_1 = np.isnan(data).sum() / data.shape[0] / data.shape[1]
    print('Target Missing Rate: {}, Original: {}, Current: {}'.format(round(miss_rate, 2), round(rate_0, 2), round(rate_1, 2)))
