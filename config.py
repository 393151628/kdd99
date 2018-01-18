# -*- coding: utf-8 -*-
import os
import redis


ENV = 'dev'
basedir = os.path.abspath(os.path.dirname(__file__))



class DevelopConfig(object):
    TESTING = True
    DEBUG = True

    MONGODB_SETTINGS = {
            'db': 'kdd99',
            'host': '10.252.99.41',
            'port': 27017,
            'username': 'kdd99',
            'password': 'kdd99',
            }

    my_ip_list = [
        '10',
        '172',
        '192',
    ]

    # redis
    redis_host = '10.252.99.41'
    redis_port = '6379'
    redis_pool = redis.ConnectionPool(host=redis_host, port=redis_port)


configs = {
    'dev': DevelopConfig,
}
