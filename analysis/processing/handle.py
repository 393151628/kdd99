# -*- coding: utf-8 -*-
import time

import celery


@celery.task
def my_celery(msg):
    print('start-------------------------%s' % msg)
    time.sleep(10)
    print('end-------------------------')
    return 'ppppp'
