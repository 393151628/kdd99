# -*- coding: utf-8 -*-
import logging
import os
import time
from random import randint
from datetime import datetime
from pytz import timezone

from analysis.utils.ip_geo import get_geo_name_by_ip, lan_ip
from config import basedir
from ..show import show_blueprint
from flask_restful import Api, Resource
import pandas as pd

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
        count, att, eventList = [], [], []
        last_minute = time_strip - 90
        beg_time = time_strip - 30
        his_num = Flow.objects.filter(createdtime__gt=str(time_strip - time_strip % 86400),
                                      createdtime__lte=str(last_minute)).count()
        # logging.info('his_num: {}'.format(his_num))
        event_obj = Flow.objects.filter(timestamp__gt=str(last_minute), timestamp__lte=str(beg_time))
        event_all = [
            [event.id, event.dip, event.dport, event.sip, event.sport, event.error_type, event.error_per,
             event.timestamp, event.domain]
            for event in event_obj
        ]
        event_df = pd.DataFrame(event_all,
                                columns=['id', 'dip', 'dport', 'sip', 'sport', 'error_type', 'error_per', 'timestamp', 'domain'])

        event_count = event_df.groupby('timestamp')['dip'].count()

        # 实时数据
        # if len(event_count.index):
        #     beg_time = int(event_count.index[-1])
        # else:
        #     beg_time = time_strip
        # beg_time = int(event_count.index[-1])
        cur_count = event_count.reindex(index=[str(beg_time + i) for i in range(-59, 1)], fill_value=0)
        cst_tz = timezone('Asia/Shanghai')
        for i in range(-59, 1):
            val = int(cur_count[i])
            his_num = his_num + val
            count.append(his_num)
            strip_time = datetime.fromtimestamp(beg_time + i).astimezone(cst_tz).strftime('%Y-%m-%d %H:%M:%S')
            att.append({"name": strip_time, "value": [strip_time, val]})

        # 测试数据
        # for i in range(-59,1):
        #     val = randint(1,5)
        #     his_num = his_num+ val
        #     count.append(his_num)
        #     strip_time = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time_strip+i))
        #     att.append({"name": strip_time, "value": [strip_time , val]})

        event_len = len(event_all)

        def _event_level(num):
            return int(num / event_len * 100)

        level_dict = event_df.groupby('dip')['timestamp'].count().astype(int).to_dict()
        level_scr = 0
        if event_len:
            for row in event_all[-10:]:
                scr_level = _event_level(level_dict[row[1]])
                level_scr += scr_level
                eventList.append({"Tip": row[1],
                                  "Tport": row[2],
                                  "Tname": get_geo_name_by_ip(row[1]),
                                  "Sname": get_geo_name_by_ip(row[3]),
                                  "Sip": row[3],
                                  "Sport": row[4],
                                  "type": lan_ip(row[1]),
                                  "EventName": str(row[0]),
                                  "EventType": row[5],
                                  "EventDes": '{}:{} -> {}:{} {}'.format(row[3], row[4], row[1], row[2], row[8]),
                                  "EventLevel": int(scr_level),
                                  "EventProbability": row[6],
                                  "EventDate": datetime.fromtimestamp(int(row[7])).astimezone(cst_tz).strftime(
                                      '%Y-%m-%d %H:%M:%S')})
            if event_len > 10:
                for row in event_all[-100:-10]:
                    scr_level = _event_level(level_dict[row[1]])
                    level_scr += scr_level
                    eventList.append({"Tip": row[1],
                                      "Tport": row[2],
                                      "Sip": row[3],
                                      "Sport": row[4],
                                      "type": lan_ip(row[1]),
                                      "EventName": str(row[0]),
                                      "EventType": row[5],
                                      "EventDes":  '{}:{} -> {}:{} {}'.format(row[3], row[4], row[1], row[2], row[8]),
                                      "EventLevel": int(scr_level),
                                      "EventProbability": row[6],
                                      "EventDate": datetime.fromtimestamp(int(row[7])).astimezone(cst_tz).strftime(
                                          '%Y-%m-%d %H:%M:%S')})

        res = {'count': count,
               'att': att,
               'eventList': eventList,
               'level': int(level_scr / (len(event_all[-100:]) + 5))
               }

        return res


api.add_resource(TestGeo, '/testgeo')
api.add_resource(MyTest, '/mytest')
api.add_resource(AbnormalEvent, '/abnormal')
