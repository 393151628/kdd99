# -*- coding: utf-8 -*-
import logging
import os
import time

from analysis.utils.ip_geo import get_geo_name_by_ip
from config import basedir
from ..show import show_blueprint
from flask_restful import Api, Resource
import pandas as pd
import json

from analysis.models import Flow

api = Api(show_blueprint)


class TestGeo(Resource):
    def get(self):
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
        logging.info('******************')
        return 'ddddddddddd'

class AbnormalEvent(Resource):
    def get(self):
        logging.info(time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())) + ' : get abnormal event.')
        time_strip = int(time.mktime(time.localtime()))
        # his_num = Flow.objects.filter(timestamp__gt=str(time_strip-time_strip%86400),timestamp__lte=str(time_strip-60)).count()
        his_num = Flow.objects.filter(timestamp__gt=str(time_strip - time_strip % 26400),
                                      timestamp__lte=str(time_strip - 60)).count()
        logging.info('his_num: {}'.format(his_num))
        event_obj = Flow.objects.filter(timestamp__gt=str(time_strip-26400),timestamp__lte=str(time_strip)).limit(100)
        event_all = [[event.id, event.dip, event.dport, event.sip, event.sport, event.error_type, event.error_per, event.timestamp] for event in event_obj]
        event_df = pd.DataFrame(event_all, columns=['id', 'dip', 'dport', 'sip', 'sport', 'error_type', 'error_per', 'timestamp'])

        his_count = pd.DataFrame([[int(time_strip) + i, 0] for i in range(60)], columns=['time', 'value'])
        event_count = event_df.groupby('timestamp')['dip'].count()
        event_count = event_count.reset_index()
        event_count = event_count.rename(columns={'timestamp': 'time', 'dip': 'value'})
        # event_count = pd.concat([event_count, his_count]).astype(int)
        #event_count测试
        event_count = event_count.astype(int)

        count = list(event_count['value'].cumsum()+his_num)[-60:]

        event_count['time'] = event_count['time'].apply(lambda x: time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(x)))
        att = [{"name": row['time'], "value": [row['time'], row['value']]} for row in
               event_count.to_dict(orient='records')[-60:]]

        event_len = len(event_all)
        def _event_level(num):
            return int(num / event_len * 100)

        level_dict = event_df.groupby('dip')['timestamp'].count().astype(int).to_dict()
        if len(event_count):
            eventList = [{"Tip": row[1],
                          "Tport": row[2],
                          "Tname": get_geo_name_by_ip(row[1]),
                          "Sname": get_geo_name_by_ip(row[3]),
                          "Sip": row[3],
                          "Sport": row[4],
                          "type": 0,
                          "EventName": str(row[0]),
                          "EventType": row[5],
                          "EventDes": '{}({}) -> {}({}):{}'.format(row[1], row[2], row[3], row[4], row[5]),
                          "EventLevel": _event_level(level_dict[row[1]]),
                          "EventProbability": row[6],
                          "EventDate": time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(int(row[7])))} for row in event_all[-10:]]
            if len(event_count)>10:
                eventList = eventList + [{"Tip": row[1],
                          "Tport": row[2],
                          "Sip": row[3],
                          "Sport": row[4],
                          "type": 0,
                          "EventName": str(row[0]),
                          "EventType": row[5],
                          "EventDes": '{}({}) -> {}({}):{}'.format(row[1], row[2], row[3], row[4], row[5]),
                          "EventLevel": _event_level(level_dict[row[1]]),
                          "EventProbability": row[6],
                          "EventDate": time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(int(row[7])))} for row in event_all[-100:-10]

                ]
        else: eventList = []

        res = {'count' : count,
               'att' : att,
               'eventList' : eventList
               }

        return res


api.add_resource(TestGeo, '/testgeo')
api.add_resource(MyTest, '/mytest')
api.add_resource(AbnormalEvent, '/abnormal')