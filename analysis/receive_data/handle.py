# -*- coding: utf-8 -*-
import threading

td_lock = threading.Lock()


def lock(func):
    def __wapper__(*args, **kwargs):
        td_lock.acquire()
        res = func(*args, **kwargs)
        td_lock.release()
        return res

    return __wapper__


class SingletonQueue(object):
    __instance = None
    timestamp_current = 0
    timestamp_last = 0
    next_con = []
    current_con = []
    last_con = []

    def __init__(self):
        pass

    def __new__(cls, *args, **kwd):
        if SingletonQueue.__instance is None:
            SingletonQueue.__instance = object.__new__(cls, *args, **kwd)
        return SingletonQueue.__instance

    @lock
    def push_current_con(self, data):
        self.current_con = self.current_con + data

    @lock
    def push_next_con(self, data):
        self.next_con = self.next_con + data

    @lock
    def push_last_con(self, data):
        self.last_con = self.last_con + data

    @lock
    def get_queue(self):
        if self.current_con:
            queue = [self.last_con, self.current_con]
            self.last_con, self.current_con = self.current_con, self.next_con
            self.timestamp_last, self.timestamp_current = self.timestamp_current, self.timestamp_current + 1
        else:
            queue = []
        return queue

    @lock
    def clean_all(self):
        self.last_con = self.current_con = self.next_con = []
        self.timestamp_last = self.timestamp_current = 0

    @lock
    def set_timestamp_current(self, timestamp):
        self.timestamp_current = timestamp

    @lock
    def set_timestamp_last(self, timestamp):
        self.timestamp_last = timestamp
