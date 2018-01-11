import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import threading
from sklearn import metrics
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Hyperparameters
L1_HIDDEN = 150
L2_HIDDEN = 100
L3_HIDDEN = 50
L4_HIDDEN = 25
L5_HIDDEN = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_one_hot(df, column):
    """
        One hot encoding for DataFrame categorical column
    """
    dummies = pd.get_dummies(df[column])

    # noinspection PyUnresolvedReferences
    for x in dummies.columns:
        df["{0}.{1}".format(column, x)] = dummies[x]
    df.drop(column, axis=1, inplace=True)

def encode_categorical_index(df, column):
    """
        Transform a categorical value in indexes (for labels)
    """
    encoder = preprocessing.LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    return encoder.classes_

def encode_continuous_zscore(df, column):
    """
        Enconde continuous columns as zscores
    """
    df[column] = (df[column] - df[column].mean()) / df[column].std()

def equalize_columns(df):
    """
        Insert missing columns from df1 on df2
    """

    #删除字段多余属性
    model_columns = ['duration', 'src_bytes', 'dst_bytes', 'count', 'serror_rate',
       'dst_host_count', 'dst_host_same_src_port_rate', 'dst_host_serror_rate',
       'outcome', 'protocol_type.tcp', 'protocol_type.udp', 'flag.OTH',
       'flag.RSTR', 'flag.S0', 'flag.S1', 'flag.S2', 'flag.S3', 'flag.SF',
       'land.0', 'land.1']
    # print(df.columns)
    for column in model_columns:
        if column not in df.columns:
            df[column] = pd.Series(0.5,
                                    index=df.index)

    return df

def to_xy(df, target):
    """
        处理成tensor张量
    """
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if \
        hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TF likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.values.astype(
            np.float32), dummies.values.astype(np.float32)
    else:
        # Regression
        return df.values.astype(np.float32), df.as_matrix(
            [target]).astype(np.float32)

def check_res(res):
    res_count = {}
    final_res = []
    for item in res:
        dip = item["content"][0]
        if res_count.get(dip):
            res_count[dip].append(item)
        else:
            res_count[dip] = [item]
    for i in res_count:
        if len(res_count[i])>1:
            final_res.extend(res_count[i])
    return final_res

def load_model(model_name):
    model = tf.contrib.keras.models.Sequential()
    model.add(tf.contrib.keras.layers.Dense(L1_HIDDEN,
                                            input_dim=20,
                                            kernel_initializer='normal',
                                            activation='relu'))
    model.add(tf.contrib.keras.layers.Dropout(0.3))
    model.add(tf.contrib.keras.layers.Dense(L2_HIDDEN,
                                            activation='relu'))
    model.add(tf.contrib.keras.layers.Dropout(0.2))
    model.add(tf.contrib.keras.layers.Dense(L3_HIDDEN,
                                            activation='relu'))
    model.add(tf.contrib.keras.layers.Dense(L4_HIDDEN,
                                            activation='relu'))
    model.add(tf.contrib.keras.layers.Dropout(0.1))
    model.add(tf.contrib.keras.layers.Dense(L5_HIDDEN,
                                            activation='relu'))
    model.add(tf.contrib.keras.layers.Dense(3,
                                            activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # Early stopping for low loss, make testing easier
    monitor = tf.contrib.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       min_delta=1e-5,
                                                       patience=5,
                                                       verbose=1,
                                                       mode='auto')

    model.load_weights(model_name)

    return model


def setup_data(df_test):
    '''
    one_hot编码及归一化
    :param df_test: Dataframe.
    :return: Dataframe
    '''
    logger.info("Testing with {0} rows.".format(len(df_test)))
    df_test = df_test[df_test['flag'].isin(['S0', 'S1', 'S2', 'S3', 'OTH', 'OTH', 'RSTR', 'SF'])]
    encode_continuous_zscore(df_test, 'duration')
    encode_one_hot(df_test, 'protocol_type')
    encode_one_hot(df_test, 'flag')
    encode_continuous_zscore(df_test, 'src_bytes')
    encode_continuous_zscore(df_test, 'dst_bytes')
    encode_one_hot(df_test, 'land')
    encode_continuous_zscore(df_test, 'count')
    # encode_continuous_zscore(df_test, 'srv_count')
    encode_continuous_zscore(df_test, 'serror_rate')
    # encode_continuous_zscore(df_test, 'srv_serror_rate')
    # encode_continuous_zscore(df_test, 'same_srv_rate')
    # encode_continuous_zscore(df_test, 'diff_srv_rate')
    encode_continuous_zscore(df_test, 'dst_host_count')
    encode_continuous_zscore(df_test, 'dst_host_same_src_port_rate')
    # encode_continuous_zscore(df_test, 'dst_host_srv_diff_host_rate')
    encode_continuous_zscore(df_test, 'dst_host_serror_rate')
    # encode_continuous_zscore(df_test, 'dst_host_srv_serror_rate')

    df_test.dropna(inplace=True, axis=1)
    # print(df_test.columns)
    # Insert missing columns
    df_test = equalize_columns(df_test)


    return df_test

def two_second_count(flow):
    '''
    count in two second
    :param flow: list. contences all the records in two second
    :return: Dataframe
    '''

    tmp = []
    flag_tcp = ['S0', 'S1', 'S2', 'S3', 'OTH', 'OTH', 'RSTR', 'SF', 'REJ']
    flag_udp = ['S0', 'S1', 'SF']
    data = DataFrame(flow)

    def _flag(row):
        if row['protocol_type'] == 'tcp':
            #         print(flag_tcp[int(row['final_status'])-1])
            if row['final_status'] < 8:
                return flag_tcp[int(row['final_status']) - 1]
            else:
                return 'OTH'
        elif row['protocol_type'] == 'udp':
            if row['final_status'] < 3:
                return flag_udp[int(row['final_status']) - 1]
            else:
                return 'SF'
        else:
            raise ValueError('New status')

    for line in flow:
        if line.get('tcp') or line.get('http') or line.get('udp') or line.get('dns'):
            tmp.append(line.get('tcp') or line.get('http') or line.get('udp') or line.get('dns'))
    test = pd.concat([DataFrame(tmp), data], axis=1)

    test.rename(columns={'in_bytes': 'src_bytes', 'out_bytes': 'dst_bytes', 'type':'protocol_type'}, inplace=True)
    test['service'] = 'http'
    test = test[test['protocol_type'].isin(['tcp', 'udp'])]
    port_tmp = test['sport'] == test['dport']
    ip_tmp = test['sip'] == test['dip']
    test['land'] = (port_tmp & ip_tmp).astype(int)

    test['flag'] = test.apply(_flag, axis=1)
    test = test.loc[:,['dip', 'dport', 'duration', 'src_bytes', 'dst_bytes', 'sip', 'sport',
       'protocol_type', 'land', 'flag']]
    test = test.dropna()

    df_len = len(test)
    same_dip_count = test.groupby(['dip'])['protocol_type'].count()
    test['count'] = test.apply(lambda row: same_dip_count[row['dip']], axis=1)


    syn_test = test[(test['flag'] == 'S0') | (test['flag'] == 'S1') | (test['flag'] == 'S2') | (test['flag'] == 'S3')]
    syn_ip_count = syn_test.groupby(['dip'])['protocol_type'].count().to_dict()
    test.loc[:, 'serror_rate'] = test.apply(
        lambda row: (syn_ip_count.get(row['dip']) or 0) / same_dip_count[row['dip']], axis=1).round(3)

    return test

def hundred_count(df, first, second):
    '''
    统计前100条连接的信息
    :param df: Dataframe. 经过two_second_count处理的dataframe
    :param first: int. 前一秒的数据流连接数量
    :param second: int. 后一秒的数据流连接数量
    :return: Dataframe.
    '''

    tmp_100 = []
    if first > 100:
        begin_ind = first - 100
    else:
        begin_ind = 100
    for i in range(begin_ind, begin_ind + second):
        tmp_df = df[i:i + 100].to_dict(orient='records')
        # print(tmp_df)
        if len(tmp_df):
            same_ip, same_ip_port, same_ip_syn  = 0, 0, 0

            dip = tmp_df[-1]['dip']
            dport = tmp_df[-1]['dport']
            for j in tmp_df:
                if j['dip'] == dip:
                    same_ip += 1
                    if j['dport'] == dport:
                        same_ip_port += 1
                    elif j['flag'] in ['S0', 'S1', 'S2', 'S3']:
                        same_ip_syn += 1
            tmp_100.append(
                list(tmp_df[-1].values()) + [same_ip, (same_ip - same_ip_port) / 100, same_ip_syn / same_ip])

    df_test = DataFrame(tmp_100, columns=['dip', 'dport', 'duration', 'src_bytes', 'dst_bytes', 'sip', 'sport',
                                        'protocol_type','land', 'flag', 'count', 'serror_rate','dst_host_count',
                                        'dst_host_same_src_port_rate', 'dst_host_serror_rate']).round(3)
    df_test = df_test.drop_duplicates(['dip',  'dport', 'sip'])
    post_info = df_test[['dip', 'dport', 'sip', 'sport']].values.tolist()

    return df_test.loc[:, ['duration', 'protocol_type', 'flag', 'src_bytes', 'dst_bytes',
                           'land', 'count', 'serror_rate', 'dst_host_count', 'dst_host_same_src_port_rate',
                           'dst_host_serror_rate']], post_info


def make_data(flow, flow_first_len, flow_second_len):
    '''
    把数据流处理成模型需要的数据格式。
    :param flow: list. 长度为2的列表，分别为紧邻的两秒的数据流。
    :return: Dataframe
    '''
    flow_one = flow[0]+flow[1]
    df_test = two_second_count(flow_one)
    df_test, post_info = hundred_count(df_test, flow_first_len, flow_second_len)

    return df_test, post_info

def main(flow,model):
    '''
    模型主控程序
    :param flow: list.
    :param model:
    :param df_train:
    :return: list. [{'content':['dip', 'dport', 'sip', 'sport'], 'error_type':error_type}, ...]
    '''
    # model = load_model('model_test_1.1.h5')
    flow_first_len = len(flow[0])
    flow_second_len = len(flow[1])
    if flow_second_len + flow_first_len > 100:
        probe_ts = flow[1][0]["probe_ts"]
        df_test, post_info = make_data(flow, flow_first_len, flow_second_len)
        df_test = setup_data(df_test)
        x_test, y_test = to_xy(df_test, 'outcome')
        pred = model.predict(x_test)*0.95
        pred_max = np.argmax(pred, axis=1)
        res = []
        error_type = {0: 'dos', 2: 'probe'}
        for i in range(len(pred_max)):
            if pred_max[i] in [0, 2] and pred[i][pred_max[i]] > 0.8:
                res.append({'content': post_info[i] + [probe_ts],
                            'error_type': [error_type[pred_max[i]], pred[i][pred_max[i]]]})
        # print(res)
        if len(res)>8:
            # print(res)
            return check_res(res)
    return []

def test_file(file_name):
    '''
    直接从接受的NPM测试文件中读入测试数据
    :param file_name: str. 测试文件的文件名
    :return:
    '''
    with open(file_name, 'r') as f:
        model = load_model('model_test_1.1.h5')
        for i in range(3):
            print('###### {} ######'.format(i))
            flow = [json.loads(f.readline().replace("'", '"').replace('u"', '"')) for line in range(2)]
            pred = main(flow, model)
            if pred:
                pred = check_res(pred)
                # print(pred)


if __name__=='__main__':
    test_file('npm_test_ts_all.txt')