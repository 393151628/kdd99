# -*- coding: utf-8 -*-
import logging
import socket
import struct
import time
import os
import datetime
import numpy as np
from celery import Celery

from analysis import create_app
from analysis.models import Flow
from analysis_machine import main, load_model, build_model
from celery import platforms
from config import ENV

platforms.C_FORCE_ROOT = True

app = create_app(ENV, name=__name__)
app.config['CELERY_BROKER_URL'] = 'redis://10.252.99.41:6379/0'


# app.config['CELERY_RESULT_BACKEND'] = 'redis://10.252.99.41:6379/0'


def make_celery(_app):
    _celery = Celery(_app.name,
                     broker=_app.config['CELERY_BROKER_URL']
                     # backend=_app.config['CELERY_RESULT_BACKEND']
                     )

    # TaskBase = _celery.Task
    #
    # class ContextTask(TaskBase):
    #     abstract = True
    #
    #     def __call__(self, *args, **kwargs):
    #         with app.app_context():
    #             return TaskBase.__call__(self, *args, **kwargs)
    #
    # _celery.Task = ContextTask
    _celery.config_from_object(app.config)

    return _celery


celery = make_celery(app)


def find_first_timestamp(data, timestamp, idx=0):
    length = len(data)
    if length == 1:
        return idx
    elif length == 0:
        return idx + 1
    else:
        i = int(length / 2)
        if data[i]['probe_ts'] == timestamp:
            return find_first_timestamp(data[:i], timestamp, idx)
        else:
            idx = idx + i + 1
            return find_first_timestamp(data[i + 1:], timestamp, idx=idx)


def create_queue(data):
    # queue_obj = SingletonQueue()
    timestamp_min = data[0].get('probe_ts')
    timestamp_max = data[-1].get('probe_ts')
    # 默认为请求来的数据是一个由时间戳升序列表
    # 判断最大的时间戳与最小时间戳是否相等
    # 相等则表示该flow内为同一秒
    # 不等则找出不等的点进行分割
    # 认为一次请求中的连接时间戳最多相差1秒
    # diff = timestamp_min - queue_obj.timestamp_current
    if timestamp_min == timestamp_max:
        queue = [[], data]
        # if diff == 0:
        #     queue_obj.push_current_con(data)
        #
        # # 代表上一秒的链接都发完了
        # elif diff == 1:
        #     queue_obj.push_next_con(data)
        #     queue = queue_obj.get_queue()
        #
        # # 重洗队列
        # elif diff >= 2:
        #     queue_obj.clean_all()
        #     queue_obj.push_last_con(data)
        #     queue_obj.set_timestamp_last(timestamp_min)
        #     queue_obj.set_timestamp_current(timestamp_min+1)
        #
        # elif diff < 0:
        #     queue_obj.push_last_con(data)

    else:
        timestamp = timestamp_min + 1
        idx = find_first_timestamp(data, timestamp)
        data_min = data[:idx]
        data_max = data[idx:]
        queue = [data_min, data_max]
        # if diff == 0:
        #     queue_obj.push_current_con(data_min)
        #     queue_obj.push_next_con(data_max)
        #
        # # 代表上一秒的链接都发完了
        # elif diff == 1:
        #     queue_obj.push_next_con(data_min)
        #     queue = queue_obj.get_queue()
        #
        # # 重洗队列
        # elif diff >= 2:
        #     queue_obj.clean_all()
        #     queue_obj.push_last_con(data_min)
        #     queue_obj.push_current_con(data_max)
        #     queue_obj.set_timestamp_last(timestamp_min)
        #     queue_obj.set_timestamp_current(timestamp)
        # elif diff < 0:
        #     queue_obj.push_last_con(data_min)
        #     queue_obj.push_current_con(data_max)

    # queue = queue_obj.get_queue()
    return queue


class SingletonModel(object):
    __instance = None
    model = ''

    def __init__(self):
        pass

    def __new__(cls, *args, **kwd):
        if SingletonModel.__instance is None:
            SingletonModel.__instance = object.__new__(cls, *args, **kwd)
            basedir = os.path.abspath(os.path.dirname(__file__))
            model_name = 'model_ip_port.h5'
            cls.model = load_model(os.path.join(basedir, 'analysis', 'utils', model_name))
            cls.model.predict(np.zeros((1, 25)))
        return SingletonModel.__instance


class SingletonDGAModel(object):
    __instance = None
    model = ''
    max_features = 38
    maxlen = 53

    def __init__(self):
        pass

    def __new__(cls, *args, **kwd):
        if SingletonDGAModel.__instance is None:
            SingletonDGAModel.__instance = object.__new__(cls, *args, **kwd)
            basedir = os.path.abspath(os.path.dirname(__file__))
            model_name = 'lstm_model.h5'
            cls.model = build_model(cls.max_features, cls.maxlen,
                                    os.path.join(basedir, 'analysis', 'utils', model_name))
            cls.model.predict([[16 for i in range(53)]])

        return SingletonDGAModel.__instance


@celery.task
def my_celery(data):
    m = SingletonModel()
    dga = SingletonDGAModel()
    model = m.model
    dga_model = dga.dga_model
    # logging.info('receive data numbers1111111111111: {0}'.format(len(data)))
    queue = data
    if queue:
        logging.info('******{0}, {1}, **********'.format(len(queue[0]), len(queue[1])))
        start = datetime.datetime.now()
        res = main(queue, model, dga_model)
        # model.predict(np.zeros((1, 25)))
        end = datetime.datetime.now()
        a = (end - start).seconds
        logging.info('运行时间:{0}'.format(a))
        for error_con in res:
            dip = error_con['content'][0]
            dport = error_con['content'][1]
            sip = error_con['content'][2]
            sport = error_con['content'][3]
            timestamp = error_con['content'][4]
            error_type = error_con['error_type']
            kwargs = {
                'dip': socket.inet_ntoa(struct.pack('I', socket.htonl(int(dip)))),
                'dport': str(int(dport)),
                'sip': socket.inet_ntoa(struct.pack('I', socket.htonl(int(sip)))),
                'sport': str(int(sport)),
                'error_type': error_type[0],
                'error_per': str(error_type[1]),
                'timestamp': str(int(timestamp)),
                'createdtime': str(int(time.time())),
            }
            Flow.objects.create(**kwargs)
