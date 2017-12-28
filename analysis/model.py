# -*- coding: utf-8 -*-
from flask_mongoengine import MongoEngine

db = MongoEngine()


class Flow(db.Document):



    deploy_id = db.StringField(required=True, unique=True)
    initiator = db.StringField()
    project_id = db.StringField()
    project_name = db.StringField()
    resource_id = db.StringField()
    deploy_name = db.StringField(required=True, unique_with='resource_id')
    resource_name = db.StringField()
    created_time = db.DateTimeField(default=datetime.datetime.now())
    environment = db.StringField()
    release_notes = db.StringField()
    mysql_tag = db.StringField()
    mysql_context = db.StringField()
    redis_tag = db.StringField()
    redis_context = db.StringField()
    mongodb_tag = db.StringField()
    mongodb_context = db.StringField()
    app_image = db.StringField()
    deploy_result = db.StringField()
    user_id = db.StringField()
    database_password = db.StringField()  # 数据库用户的密码
    apply_status = db.StringField()  # 部署申请状态
    approve_status = db.StringField()  # 部署审批状态
    approve_suggestion = db.StringField()  # 审批意见
    deploy_type = db.StringField()  # 部署类型
    disconf_list = db.ListField(db.EmbeddedDocumentField('DisconfIns'))
    is_deleted = db.IntField(required=False, default=0)
    is_rollback = db.IntField(required=False, default=0)
    deleted_time = db.DateTimeField(default=datetime.datetime.now())
    capacity_info = db.StringField(required=False, default="{}")
    department = db.StringField()

    meta = {
        'collection': 'deployment',
        'index': [
            {
                'fields': ['deploy_id', 'deploy_name'],
                'sparse': True,
                }
            ],
        'index_background': True
    }
