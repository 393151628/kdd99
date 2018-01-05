# -*- coding: utf-8 -*-
import json

from flask_celery import my_celery
from analysis.receive_data import receive_blueprint
from flask import request
from flask_restful import Api, Resource

api = Api(receive_blueprint)


class ReciveData(Resource):
    def post(self):
        data = request.get_data()
        # data = json.loads(data)
        with open('/tmp/data', 'rb') as f:
            f.write(data)
        # task = my_celery.apply_async(args=[data])
        return 'success'


api.add_resource(ReciveData, '/')
