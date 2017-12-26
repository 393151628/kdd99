# -*- coding: utf-8 -*-
import json

from config import SingletonQueue
from ..receive_data import receive_blueprint
from flask import request
from flask_restful import Api, Resource

receive = Api(receive_blueprint)


def create_queue(data):
    queue_obj = SingletonQueue()
    queue = queue_obj.get_queue(data)
    if queue == 1:
        return None
    else:
        return queue


class ReciveData(Resource):
    def post(self):
        data = request.get_data()
        data = json.loads(data)
        queue = create_queue(data)
        # TODO: 将接收的数据发送给转化接口
        return 'success'


receive.add_resource(ReciveData, '/')
