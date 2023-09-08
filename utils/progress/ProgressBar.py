import sys
from datetime import datetime, timedelta
import numpy as np


class ProgressBar:
    def __init__(self, total_counts, total_bar_length=30, display_counter=True, display_timing=True, refresh_rate=1, front_message='[INFO] Progress: '):
        self.total_counts = total_counts
        self._current_counts = 0
        self.total_bar_length = total_bar_length
        self._current_bar_length = 0
        self._current_bar_string_length = 0
        self._display_counter = display_counter
        self._display_timing = display_timing
        self._refresh_rate = timedelta(milliseconds=int(1000 * refresh_rate))
        self._init_time = datetime.now()
        self._last_refresh = self._init_time
        self._front_message = front_message
        # print empty bar
        self.step(0, initial=True)

    def step(self, count=1, initial=False):
        now = datetime.now()
        self._current_counts += count
        if self._current_counts > self.total_counts:
            self._current_counts = self.total_counts
        final_update = self._current_counts == self.total_counts
        progress = self._current_counts / self.total_counts
        new_bar_length = int(self.total_bar_length * progress)
        update_required = (new_bar_length > self._current_bar_length) or self._display_counter or self._display_timing
        regular_update = (now - self._last_refresh > self._refresh_rate) and update_required
        if regular_update or final_update or initial:
            bar_string = '\r'
            bar_string += self._front_message
            if not self._front_message[-1] == ' ':
                bar_string += ' '
            bar_string += '[{}{}{}] {:>3d}%'.format(
                '=' * max(new_bar_length - 1, 0),
                '=' if final_update else ('>' * int(new_bar_length > 0)),
                '.' * (self.total_bar_length - new_bar_length),
                int(progress * 100),
            )
            if self._display_counter:
                total_counts = str(self.total_counts)
                current_counts = str(self._current_counts).rjust(len(total_counts))
                bar_string += ' ({}/{})'.format(current_counts, self.total_counts)
            if self._display_timing and self._current_counts > 0:
                time_difference = (now - self._init_time).total_seconds()
                if not final_update:
                    time_difference = (1 - progress) / progress * time_difference
                seconds = int(np.round(time_difference))
                minutes = seconds // 60
                seconds = seconds - 60 * minutes
                hours = minutes // 60
                minutes = minutes - 60 * hours
                time_string = ' {:04d}h:{:02d}m:{:02d}s'.format(hours, minutes, seconds)
                if not final_update:
                    bar_string += time_string + ' remaining.'
                else:
                    bar_string += ' Total run time was' + time_string
            new_bar_string_length = len(bar_string)
            bar_string = bar_string.ljust(max(new_bar_string_length, self._current_bar_string_length))
            if final_update:
                bar_string += '\n'
            sys.stdout.write(bar_string)
            sys.stdout.flush()
            self._current_bar_length = new_bar_length
            self._current_bar_string_length = new_bar_string_length
            self._last_refresh = datetime.now()


class ProgressIterator(object):

    def __init__(self, iterable, total_counts=None, pbar_kws=None):
        self.iterable = iterable
        if total_counts is not None:
            self._total_counts = total_counts
        elif hasattr(iterable, '__len__'):
            self._total_counts = len(iterable)
        else:
            raise Exception('[ERROR] Unable to determine total counts from given iterable')
        self._pbar_kws = pbar_kws if pbar_kws is not None else {}
        self._pbar = None
        self._iterator = None

    def __iter__(self):
        self._pbar = ProgressBar(self._total_counts, **self._pbar_kws)
        self._iterator = iter(self.iterable)
        return self

    def __next__(self):
        output = next(self._iterator)
        self._pbar.step()
        return output