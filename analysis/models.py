# -*- coding: utf-8 -*-
from datetime import datetime
from flask_mongoengine import MongoEngine

db = MongoEngine()


class Flow(db.Document):
    dip = db.StringField()
    dport = db.StringField()
    sip = db.StringField()
    sport = db.StringField()
    error_type = db.StringField()
    error_per = db.StringField()
    timestamp = db.StringField()
    created_time = db.DateTimeField(auto_now_add=True, default=datetime.now())

    meta = {
        'collection': 'flow',
        'index_background': True
    }
