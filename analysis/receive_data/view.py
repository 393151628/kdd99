# -*- coding: utf-8 -*-
import json

from flask_celery import my_celery
from analysis.receive_data import receive_blueprint
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


class ReciveData(Resource):
    def post(self):
        data = request.get_data()
        data = json.loads(data)
        task = my_celery.apply_async(args=[data])
        return 'success'


api.add_resource(ReciveData, '/')
