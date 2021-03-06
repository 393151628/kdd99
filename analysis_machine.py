#增加srv_serror_rate


import os
import json
import logging
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tldextract
from pandas import DataFrame
from sklearn import preprocessing
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

# Hyperparameters
L1_HIDDEN = 375
L2_HIDDEN = 200
L3_HIDDEN = 150
L4_HIDDEN = 75
L5_HIDDEN = 50
CONN_LEN = 0
DDOS_LEN = 0
SCAN_LEN = 0
model_name = 'lstm_model.h5'

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
    model_columns = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
       'serror_rate', 'srv_serror_rate', 'same_srv_rate', 'diff_srv_rate',
       'dst_host_count', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'outcome', 'protocol_type.tcp',
       'protocol_type.udp', 'flag.OTH', 'flag.RSTR', 'flag.S0', 'flag.S1',
       'flag.S2', 'flag.S3', 'flag.SF', 'land.0', 'land.1']
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
    dport_count = {}
    sport_count = {}
    final_res = []
    for item in res:
        dip = item["content"][0]
        dport  = item["content"][1]
        sip = item["content"][2]
        sport = item["content"][3]
        if dport_count.get(dip):
            if dport_count[dip].get(sip):
                if dport_count[dip][sip].get(dport):
                    dport_count[dip][sip][dport].append(item)
                else: dport_count[dip][sip][dport] = [item]
                if sport_count[dip][sip].get(sport):
                    sport_count[dip][sip][sport].append(item)
                else: sport_count[dip][sip][sport] = [item]
            else :
                dport_count[dip][sip] = {dport: [item]}
                sport_count[dip][sip] = {sport: [item]}
        else:
            dport_count[dip] = {sip: {dport: [item]}}
            sport_count[dip] = {sip: {sport: [item]}}
    for dip in dport_count:
        for sip in dport_count[dip]:
            if len(dport_count[dip][sip]) > len(sport_count[dip][sip]):
                if len(dport_count[dip][sip]) > 10:
                    for dport in dport_count[dip][sip]:
                        for k in dport_count[dip][sip][dport]:
                            final_res.append({'content': k['content'],
                                              'error_type': ['scan', 0.98],
                                              'domain': ''})
                elif len(dport_count[dip][sip]) > 5:
                    for dport in dport_count[dip][sip]:
                        for k in dport_count[dip][sip][dport]:
                            final_res.append({'content': k['content'],
                                              'error_type': ['scan', k['error_type'][1]],
                                              'domain': ''})
            elif len(dport_count[dip][sip]) < len(sport_count[dip][sip]):
                if  len(sport_count[dip][sip]) > 10:
                    for sport in sport_count[dip][sip]:
                        for k in sport_count[dip][sip][sport]:
                            final_res.append({'content': k['content'],
                                              'error_type': ['ddos', 0.98],
                                              'domain': ''})
                elif  len(sport_count[dip][sip]) > 5:
                    for sport in sport_count[dip][sip]:
                        for k in sport_count[dip][sip][sport]:
                            final_res.append({'content': k['content'],
                                              'error_type': ['ddos', k['error_type'][1]],
                                              'domain': ''})
            elif len(dport_count[dip][sip]) > 15:
                    for dport in dport_count[dip][sip]:
                        for k in dport_count[dip][sip][dport]:
                            final_res.append(k)

    return final_res

def load_model(model_name):
    model = tf.contrib.keras.models.Sequential()
    model.add(tf.contrib.keras.layers.Dense(L1_HIDDEN,
                                            input_dim=26,
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
    global CONN_LEN
    logger.info("Testing with {0} rows.".format(len(df_test)))
    CONN_LEN += len(df_test)
    df_test = df_test[df_test['flag'].isin(['S0', 'S1', 'S2', 'S3', 'OTH', 'OTH', 'RSTR', 'SF'])]
    encode_continuous_zscore(df_test, 'duration')
    encode_one_hot(df_test, 'protocol_type')
    encode_one_hot(df_test, 'flag')
    encode_continuous_zscore(df_test, 'src_bytes')
    encode_continuous_zscore(df_test, 'dst_bytes')
    encode_one_hot(df_test, 'land')
    encode_continuous_zscore(df_test, 'count')
    encode_continuous_zscore(df_test, 'srv_count')
    encode_continuous_zscore(df_test, 'serror_rate')
    encode_continuous_zscore(df_test, 'srv_serror_rate')
    encode_continuous_zscore(df_test, 'same_srv_rate')
    encode_continuous_zscore(df_test, 'diff_srv_rate')
    encode_continuous_zscore(df_test, 'dst_host_count')
    # encode_continuous_zscore(df_test, 'dst_host_srv_count')
    # encode_continuous_zscore(df_test, 'dst_host_same_srv_rate')
    # encode_continuous_zscore(df_test, 'dst_host_diff_srv_rate')
    encode_continuous_zscore(df_test, 'dst_host_same_src_port_rate')
    encode_continuous_zscore(df_test, 'dst_host_srv_diff_host_rate')
    encode_continuous_zscore(df_test, 'dst_host_serror_rate')
    encode_continuous_zscore(df_test, 'dst_host_srv_serror_rate')

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
            pro = line.get('tcp') or line.get('http') or line.get('udp') or line.get('dns')
            if pro:
                pro['type'] = line['type']
                tmp.append(pro)
    test = DataFrame(tmp)

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
    same_dport_count = test.groupby(['dip', 'dport'])['protocol_type'].count()
    same_tmp = (same_dport_count / df_len).round(3)

    #same host count
    test['count'] = test.apply(lambda row: same_dip_count[row['dip']], axis=1)

    #same host srv count
    test['srv_count'] = test.apply(lambda row: same_dport_count[row['dip']][row['dport']], axis=1)

    test['same_srv_rate'] = test.apply(lambda row: same_tmp[row['dip']][row['dport']], axis=1)
    test['diff_srv_rate'] = test.apply(
        lambda row: (same_dip_count[row['dip']] - same_dport_count[row['dip']][row['dport']]) / df_len, axis=1).round(3)

    #same host SYN error rate
    syn_test = test[(test['flag'] == 'S0') | (test['flag'] == 'S1') | (test['flag'] == 'S2') | (test['flag'] == 'S3')]
    syn_ip_count = syn_test.groupby(['dip'])['protocol_type'].count().to_dict()
    same_dport_count = test.groupby(['dip', 'dport'])['protocol_type'].count()
    same_tmp = (same_dport_count / df_len).round(3)

    test.loc[:, 'serror_rate'] = test.apply(
        lambda row: (syn_ip_count.get(row['dip']) or 0) / same_dip_count[row['dip']], axis=1).round(3)

    #same host server SYN error rate
    syn_port_count = syn_test.groupby(['dip', 'dport'])['protocol_type'].count().to_dict()
    test.loc[:, 'srv_serror_rate'] = test.apply(lambda row: (syn_port_count.get((row['dip'], row['dport'])) or 0) /
                                                            same_dport_count[row['dip']][row['dport']], axis=1).round(3)

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
        begin_ind = first
    else:
        begin_ind = 100
    for i in range(begin_ind, first + second):
        tmp_df = df[i-100 :i].to_dict(orient='records')
        # print(tmp_df)
        if len(tmp_df):
            same_ip, same_dip_dport, same_dip_sport, same_ip_syn, same_port_syn, same_dip_port_diff_sip  = 0, 0, 0, 0, 0, 0

            dip = tmp_df[-1]['dip']
            sport = tmp_df[-1]['sport']
            sip = tmp_df[-1]['sip']
            dport = tmp_df[-1]['dport']
            #     print(dip, dport, sip)
            for j in tmp_df:
                if j['dip'] == dip:
                    same_ip += 1
                    if j['dport'] == dport:
                        same_dip_dport += 1
                        if j['flag'] in ['S0', 'S1', 'S2', 'S3']:
                            same_port_syn += 1
                        if j['sip'] != sip:
                            same_dip_port_diff_sip += 1
                    if j['sport'] == sport:
                        same_dip_sport += 1


                    elif j['flag'] in ['S0', 'S1', 'S2', 'S3']:
                        same_ip_syn += 1
            tmp_100.append(
                list(tmp_df[-1].values()) + [same_ip, same_dip_sport / 100,same_dip_port_diff_sip / same_dip_dport,
                                             same_ip_syn / same_ip, same_port_syn / same_dip_dport])

    df_test = DataFrame(tmp_100, columns=['dip', 'dport', 'duration', 'src_bytes', 'dst_bytes', 'sip', 'sport',
                                          'protocol_type', 'land', 'flag', 'count', 'srv_count', 'same_srv_rate',
                                          'diff_srv_rate', 'serror_rate', 'srv_serror_rate', 'dst_host_count',
                                          'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                                          'dst_host_serror_rate', 'dst_host_srv_serror_rate']).round(3)
    df_test = df_test.drop_duplicates(['dip', 'dport', 'sip', 'sport'])
    post_info = df_test[['dip', 'dport', 'sip', 'sport']].values.tolist()

    return df_test.loc[:, ['duration', 'protocol_type', 'flag', 'src_bytes', 'dst_bytes',
                           'land', 'count', 'srv_count', 'same_srv_rate', 'diff_srv_rate',
                           'serror_rate', 'srv_serror_rate', 'dst_host_count',
                           'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                           'dst_host_serror_rate', 'dst_host_srv_serror_rate']], post_info

    # return df_test.loc[:, ['duration', 'protocol_type', 'flag', 'src_bytes', 'dst_bytes',
    #                        'land', 'count', 'srv_count', 'same_srv_rate','diff_srv_rate',
    #                        'serror_rate', 'srv_serror_rate', 'dst_host_count',
    #                        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    #                        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    #                        'dst_host_serror_rate', 'dst_host_srv_serror_rate']], post_info


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

def build_model(max_features, maxlen, dga_model_name):
    """Build LSTM model"""
    dga_model = Sequential()
    dga_model.add(Embedding(max_features, 128, input_length=maxlen))
    dga_model.add(LSTM(128))
    dga_model.add(Dropout(0.5))
    dga_model.add(Dense(1))
    dga_model.add(Activation('sigmoid'))

    dga_model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')

    dga_model.load_weights(dga_model_name)

    return dga_model

def dga_train(dns_test, dga_model):
    valid_chars = {'7': 1, '4': 2, 'b': 3, '9': 4, '5': 5, 'k': 6, 'o': 7, 'v': 8, 'h': 9, 'i': 10, '8': 11, 'y': 12, '2': 13,
                   '1': 14, '6': 15, '-': 16, 'u': 17, 'g': 18, 'w': 19, '3': 20, 'q': 21, 'z': 22, 'x': 23, 's': 24, 'a': 25,
                   't': 26, 'l': 27, 'p': 28, 'e': 29, 'm': 30, 'n': 31, 'f': 32, '0': 33, 'c': 34, 'r': 35, 'j': 36, 'd': 37}

    X = [[valid_chars[y] for y in x if valid_chars.get(y)] for x in dns_test]
    X = sequence.pad_sequences(X, maxlen=53)
    probs = dga_model.predict_proba(X)
    return probs

def dga_check(flow, dga_model):
    flow = flow[0]+flow[1]
    dns_test, dns_check, dns_info, dga_res = [], [], [], []
    for event in flow:
        if event['type'] == 'dns':
            domain = tldextract.extract(event['dns']['req_name']).domain
            if domain not in ['wxjuyun']:
                dns_test.append(domain)
                dns_info.append({'content': [event['dns']['dip'], event['dns']['dport'], event['dns']['sip'],
                                            event['dns']['sport'], event['probe_ts']],
                                 'domain': domain})
    if dns_test:
        dns_check = dga_train(dns_test, dga_model)
    for ival, val in enumerate(dns_check):
        if val[0] > 0.8:
            dga_res.append({'content': dns_info[ival]['content'],
                            'error_type': ['DGA', round(val[0], 3)],
                            'domain': dns_info[ival]['domain']})
    return dga_res

def main(flow, model, dga_model):
    '''
    模型主控程序
    :param flow: list.
    :param model:
    :param df_train:
    :return: list. [{'content':['dip', 'dport', 'sip', 'sport'], 'error_type':error_type, 'domain':domain}, ...]
    '''
    global DDOS_LEN, SCAN_LEN
    flow_first_len = len(flow[0])
    flow_second_len = len(flow[1])
    res = []
    if flow_second_len + flow_first_len > 100:
        probe_ts = flow[1][0]["probe_ts"]
        df_test, post_info = make_data(flow, flow_first_len, flow_second_len)
        df_test = setup_data(df_test)
        x_test, y_test = to_xy(df_test, 'outcome')
        pred = model.predict(x_test)
        pred_max = np.argmax(pred, axis=1)
        error_type = {0: 'ddos', 2: 'probe', 1: 'normal'}
        seed = random.uniform(0.9, 0.96)
        tmp_res = []
        for i in range(len(pred_max)):
            #攻击流量测试
            # if post_info[i][2] == 184291113:
            #     SCAN_LEN += 1
            #     print('nmap', post_info[i], probe_ts, 'error_type', [error_type[pred_max[i]], round(float(pred[i][pred_max[i]]), 3)])
            # if post_info[i][2] == 16843009:
            #     DDOS_LEN += 1
            #     print('ddos',post_info[i], probe_ts, 'error_type', [error_type[pred_max[i]], pred[i]])
            if pred_max[i] in [0, 2] and pred[i][pred_max[i]] > 0.5:
                tmp_res.append({'content': post_info[i] + [probe_ts],
                                'error_type': [error_type[pred_max[i]], round(float(pred[i][pred_max[i]] * seed), 3)],
                                'domain': ''})
        if len(tmp_res) > 5:
            res = check_res(tmp_res)
    dga_res = dga_check(flow, dga_model)
    if dga_res:
        res.extend(dga_res)
    return res

def test_file(file_name):
    '''
    直接从接受的NPM测试文件中读入测试数据
    :param file_name: str. 测试文件的文件名
    :return:
    '''
    true_nmap, false_nmap, true_ddos, false_ddos = 0,0,0,0
    max_features = 38
    maxlen = 53
    with open(file_name, 'rb') as f:
        model = load_model('model_test_all.h5')
        dga_model_name = model_name
        dga_model = build_model(max_features, maxlen, dga_model_name)
        flows = f.read().split(b'|')
        flow_len = len(flows)
        print(flow_len)
        for i in range(100):
            print('###########{}#############: {}-{}-{}'.format(i, CONN_LEN, DDOS_LEN, SCAN_LEN))
            flow = [json.loads(flows[i]), json.loads(flows[i+1])]
            pred = main(flow, model, dga_model)
            if pred:
                for i in pred:
                    if i['content'][2] == 184291113:
                        true_nmap += 1
                    elif i['content'][2] == 16843009:
                        true_ddos += 1
                    elif i['error_type'][0] == 'scan':
                        false_nmap += 1
                    else:
                        false_ddos += 1
                print(len(pred),pred)
        print(CONN_LEN, DDOS_LEN, SCAN_LEN, true_nmap, false_nmap, true_ddos, false_ddos)
        print('nmap:')
        print('recall: {}'.format(true_nmap/SCAN_LEN))
        print('precision: {}'.format(true_nmap/(true_nmap+false_nmap)))
        print('F1: {}'.format(2*true_nmap/(true_nmap+false_nmap+SCAN_LEN)))

        print('ddos:')
        print('recall: {}'.format(true_ddos / DDOS_LEN))
        print('precision: {}'.format(true_ddos / (true_ddos + false_ddos)))
        print('F1: {}'.format(2 * true_ddos / (true_ddos + false_ddos + DDOS_LEN)))

if __name__=='__main__':
    # test_file('npm_test_ts_all.txt')
    test_file('testdata_all')
    # with open('testdata', 'rb') as f:
    #     flows = f.read().split(b'|')
    #     for flow in flows:
    #         for line in json.loads(flow):
    #             flow = line.get('tcp') or line.get('http') or line.get('udp') or line.get('dns')
    #             if flow['sip'] == 184291113:
    #                 print(line)

# import cProfile
# cProfile.run("test_file('testdata')")