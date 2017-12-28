# -*- coding: utf-8 -*-
import os
from analysis.utils.ip_geo import get_geo_name_by_ip
from config import basedir
from ..show import show_blueprint
from flask import request
from flask_restful import Api, Resource

api = Api(show_blueprint)


class TestGeo(Resource):
    def get(self):
        result = []
        with open(os.path.join(basedir, 'analysis/show/geo.txt'), 'rb') as f:
            content_list = f.readlines()
            for ip in content_list:
                ip = str(ip.strip(), encoding='utf-8')
                name = get_geo_name_by_ip(ip)
                result.append(name)
        data = {
            'geo_name': result
        }
        return data


api.add_resource(TestGeo, '/testgeo')