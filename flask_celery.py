# -*- coding: utf-8 -*-
import logging
import socket
import struct
import datetime
from flask import Flask
from celery import Celery

from analysis.models import Flow
from config import model
from analysis.receive_data.view import create_queue
from analysis.utils.analysis_machine import main
from celery import platforms

platforms.C_FORCE_ROOT = True

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'


def make_celery(_app):
    _celery = Celery(_app.name,
                    broker=_app.config['CELERY_BROKER_URL'],
                    backend=_app.config['CELERY_RESULT_BACKEND']
                    )
    _celery.conf.update(_app.config)
    return _celery


celery = make_celery(app)


@celery.task
def my_celery(data):
    queue = create_queue(data)
    if queue:
        logging.info('******{0}, {1}, **********'.format(len(queue[0]), len(queue[1])))
        start = datetime.datetime.now()
        res = main(queue, model)
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
            }
            Flow.objects.create(**kwargs)
