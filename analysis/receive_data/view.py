# -*- coding: utf-8 -*-
import json
import datetime
import logging
import socket
import struct

from analysis.models import Flow
from flask_celery import my_celery
from analysis.receive_data import receive_blueprint
from flask import request
from flask_restful import Api, Resource
from flask_celery import create_queue

from analysis_machine import main

api = Api(receive_blueprint)


class ReciveData(Resource):
    def post(self):
        data = request.get_data()
        data = json.loads(data)
        task = my_celery.apply_async(args=[data])
        return 'success'

# class ReciveDataFile(Resource):
#     def post(self):
#         queue = create_queue(json.loads(request.get_data()))
#         if queue:
#             logging.info('******{0}, {1}, **********'.format(len(queue[0]), len(queue[1])))
#             start = datetime.datetime.now()
#             res = main(queue, model)
#             end = datetime.datetime.now()
#             a = (end - start).seconds
#             logging.info('运行时间:{0}'.format(a))
#             for error_con in res:
#                 dip = error_con['content'][0]
#                 dport = error_con['content'][1]
#                 sip = error_con['content'][2]
#                 sport = error_con['content'][3]
#                 timestamp = error_con['content'][4]
#                 error_type = error_con['error_type']
#                 kwargs = {
#                     'dip': socket.inet_ntoa(struct.pack('I', socket.htonl(int(dip)))),
#                     'dport': str(int(dport)),
#                     'sip': socket.inet_ntoa(struct.pack('I', socket.htonl(int(sip)))),
#                     'sport': str(int(sport)),
#                     'error_type': error_type[0],
#                     'error_per': str(error_type[1]),
#                     'timestamp': str(int(timestamp)),
#                 }
#                 Flow.objects.create(**kwargs)

api.add_resource(ReciveData, '/')
# api.add_resource(ReciveDataFile, '/ReciveDataFile')
