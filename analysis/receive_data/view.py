# -*- coding: utf-8 -*-
import json
import logging
import redis

from analysis.receive_data import receive_blueprint
from flask import request
from flask_restful import Api, Resource
from config import configs, ENV
from flask_celery import my_celery

api = Api(receive_blueprint)


class ReciveData(Resource):
    def post(self):
        data = request.get_data()
        r = redis.Redis(connection_pool=configs[ENV].redis_pool, db=1)
        if r.exists('data'):
            data1 = json.loads(r.get('data'))
            data2 = json.loads(data)
            logging.info('send celery numbers: {0}****************{1}'.format(len(data1), len(data2)))
            queue = [data1, data2]
            task = my_celery.apply_async(args=[queue])
            r.delete('data')
        else:
            r.set('data', data)
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
