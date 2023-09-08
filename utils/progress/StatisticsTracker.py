class StatisticsTracker(object):
    def __init__(self, count=0):
        self._count = count

    def get_state(self):
        raise NotImplementedError()

    def set_state(self, **kwargs):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def update(self, value, weight=1.):
        raise NotImplementedError()

    def summary(self, **kwargs):
        raise NotImplementedError()

    def count(self):
        return self._count

    def mean(self):
        raise NotImplementedError()

    def var(self, **kwargs):
        raise NotImplementedError()

    def std(self, **kwargs):
        raise NotImplementedError()
