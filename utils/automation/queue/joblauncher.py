import os
from time import sleep
from .computejob import ComputeJob
from .queuemanager import QueueManager, JobStatus


class JobLauncher(object):

    def __init__(self, *queues, wait=False, sleep_time=30):
        for queue in queues:
            assert isinstance(queue, QueueManager)
        self.queues = list(queues)
        self._active_queue = None
        self._pid = os.getpid()
        self._wait = wait
        self._sleep_time = sleep_time

    def run(self):
        while True:
            print('[INFO] Launcher {}: Reading queue...'.format(self._pid))
            response = None
            for queue in self.queues:
                response = queue.retrieve_job()
                if response is not None:
                    self._active_queue = queue
                    break
            if response is None:
                print('[INFO] Launcher {}: No jobs remaining')
                if self._wait:
                    print('[INFO] Waiting for new jobs...')
                    sleep(self._sleep_time)
                    continue
                else:
                    print('[INFO] Returning...')
                    return
            job, job_lock = response
            print('[INFO] Launcher {}: Processing job {}'.format(self._pid, job.id))
            self._active_queue.set_job_status(job.id, JobStatus.RUNNING)
            log_file_path = self._active_queue.get_log_file(job.id)
            log = job.run(log_file=log_file_path)
            if log[ComputeJob.LogKey.RETURNCODE] == 0:
                self._active_queue.set_job_status(job.id, JobStatus.FINISHED)
            else:
                self._active_queue.set_job_status(job.id, JobStatus.FAILED)
            self._active_queue.unlock_job(job_lock)
            self._active_queue = None
