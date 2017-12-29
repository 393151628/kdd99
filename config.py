# -*- coding: utf-8 -*-
import os

from analysis.utils.analysis_machine import load_model

basedir = os.path.abspath(os.path.dirname(__file__))
model = load_model(os.path.join(basedir, 'analysis', 'utils', 'ss_model_rej_20.h5'))


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


configs = {
    'dev': DevelopConfig,
}
