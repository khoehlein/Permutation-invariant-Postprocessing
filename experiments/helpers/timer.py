import time


class Timer(object):

    def __init__(self):
        self.t_start = None
        self.t_stop = None

    def start(self):
        if self.t_start is not None:
            raise Exception('[ERROR] Timer already started.')
        self.t_start = time.time()
        return self.t_start

    def stop(self):
        now = time.time()
        if not self.is_started():
            raise Exception('[ERROR] Timer stopped before it was started.')
        if self.is_stopped():
            raise Exception('[ERROR] Timer already stopped.')
        self.t_stop = now
        return now

    def is_started(self):
        return self.t_start is not None and self.t_stop is None

    def is_stopped(self):
        return self.t_start is not None and self.t_stop is not None

    def read(self):
        return self.t_stop - self.t_start

    def reset(self):
        self.t_start = None
        self.t_stop = None
        return self
