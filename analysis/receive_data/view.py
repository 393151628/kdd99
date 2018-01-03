# -*- coding: utf-8 -*-
import json
import socket
import struct
import logging

from analysis.model import Flow
from config import model
from analysis.utils.analysis_machine import main
from .handle import SingletonQueue
from ..receive_data import receive_blueprint
from flask import request
from flask_restful import Api, Resource

api = Api(receive_blueprint)


def find_first_timestamp(data, timestamp, idx=0):
    length = len(data)
    if length == 1:
        return idx
    elif length == 0:
        return idx + 1
    else:
        i = int(length/2)
        if data[i]['probe_ts'] == timestamp:
            return find_first_timestamp(data[:i], timestamp, idx)
        else:
            idx = idx + i + 1
            return find_first_timestamp(data[i+1:], timestamp, idx=idx)


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


class ReciveData(Resource):
    def post(self):
        data = request.get_data()
        data = json.loads(data)
        queue = create_queue(data)
        if queue:
            logging.info('******{0}, {1}, **********'.format(len(queue[0]), len(queue[1])))
            res = main(queue, model)
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
                }
                Flow.objects.create(**kwargs)

        return 'success'


api.add_resource(ReciveData, '/')
