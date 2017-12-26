# -*- coding: utf-8 -*-
import threading


class SingletonQueue(object):
    __instance = None
    lock = threading.Lock()
    queue = []

    def __init__(self):
        pass

    def __new__(cls, *args, **kwd):
        if SingletonQueue.__instance is None:
            SingletonQueue.__instance = object.__new__(cls, *args, **kwd)
        return SingletonQueue.__instance

    def get_queue(self, data):
        try:
            self.lock.acquire()
            self.queue.append(data)
            self.queue = self.queue[-2:]
            return self.queue
        finally:
            self.lock.release()


