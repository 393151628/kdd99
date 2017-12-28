# -*- coding: utf-8 -*-
import os

basedir = os.path.abspath(os.path.dirname(__file__))


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
