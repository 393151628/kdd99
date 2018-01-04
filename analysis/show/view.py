# -*- coding: utf-8 -*-
import os

from analysis.processing.handle import my_celery
from analysis.utils.ip_geo import get_geo_name_by_ip
from config import basedir
from ..show import show_blueprint
from flask_restful import Api, Resource

api = Api(show_blueprint)


class TestGeo(Resource):
    def get(self):
        import time
        time.sleep(20)
        result = {}
        with open(os.path.join(basedir, 'analysis/show/geo.txt'), 'rb') as f:
            content_list = f.readlines()
            for ip in content_list:
                ip = str(ip.strip(), encoding='utf-8')
                name = get_geo_name_by_ip(ip)
                result[name] = ip

        res = [{'name': k, 'ip': v} for k, v in result.items()]
        data = {
            'res': res[:10]
        }
        return data


class MyTest(Resource):
    def get(self):
        task = my_celery.delay('asdasdasd')
        return 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'


api.add_resource(TestGeo, '/testgeo')
api.add_resource(MyTest, '/mytest')
