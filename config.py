# -*- coding: utf-8 -*-
import os
import numpy as np

from analysis.utils.analysis_machine import load_model

ENV = 'dev'
basedir = os.path.abspath(os.path.dirname(__file__))
model_name = 'model_ip_port.h5'
model = load_model(os.path.join(basedir, 'analysis', 'utils', model_name))
model.predict(np.zeros((1, 25)))


class DevelopConfig(object):
    TESTING = True
    DEBUG = True

    MONGODB_SETTINGS = {
            'db': 'kdd99',
            'host': '172.28.20.124',
            'port': 27017,
            'username': 'kdd99',
            'password': 'kdd99',
            }

    my_ip_list = [
        '10',
        '172',
        '192',
    ]

    # celery
    CELERY_BROKER_URL = 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'


configs = {
    'dev': DevelopConfig,
}
