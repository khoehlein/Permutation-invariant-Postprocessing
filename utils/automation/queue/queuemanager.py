import os
import json
import fcntl
import sqlite3
from sqlite3 import Error as SQLError
from enum import Enum
from .computejob import ComputeJob
from ..helpers import get_timestamp_string


LAUNCHER_QUEUE = os.path.expanduser('~/PycharmProjects/queue')
DATABASE_FOLDER = 'sqlite'
DATABASE_FILE = 'queue.db'
DATABASE_LOCK_FILE = 'queue.lock'
CONFIG_FOLDER = 'configs'
LOG_FOLDER = 'logs'


class JobStatus(Enum):
    TBA = 'tba'
    OPEN = 'open'
    PENDING = 'pending'
    RUNNING = 'running'
    FINISHED = 'finished'
    FAILED = 'failed'


class JobTable(object):

    NAME = 'jobs'
    ID_KEY = 'id'

    class DataKey(object):
        STATUS = 'status'
        CONFIG_FILE_NAME = 'config_file_name'
        LAST_MODIFIED = 'last_modified'
        DESCRIPTION = 'description'

    @staticmethod
    def get_creation_command():
        entries = [
            '{} INTEGER PRIMARY KEY AUTOINCREMENT'.format(JobTable.ID_KEY),
            '{} TEXT NOT NULL'.format(JobTable.DataKey.STATUS),
            '{} TEXT NOT NULL'.format(JobTable.DataKey.CONFIG_FILE_NAME),
            '{} TEXT NOT NULL'.format(JobTable.DataKey.LAST_MODIFIED),
            '{} TEXT'.format(JobTable.DataKey.DESCRIPTION),
        ]
        command = 'CREATE TABLE IF NOT EXISTS {} ({});'.format(JobTable.NAME, ', '.join(entries))
        return command


class TimestampTable(object):

    NAME = 'timestamps'
    ID_KEY = 'id'
    JOB_ID_KEY = 'job_id'

    class DataKey(object):
        TIMESTAMP = 'timestamp'
        STATUS_OLD = 'status_old'
        STATUS_NEW = 'status_new'

    @staticmethod
    def get_creation_command():
        entries = [
            '{} INTEGER PRIMARY KEY AUTOINCREMENT'.format(TimestampTable.ID_KEY),
            '{} INTEGER NOT NULL'.format(TimestampTable.JOB_ID_KEY),
            '{} TEXT NOT NULL'.format(TimestampTable.DataKey.TIMESTAMP),
            '{} TEXT NOT NULL'.format(TimestampTable.DataKey.STATUS_OLD),
            '{} TEXT NOT NULL'.format(TimestampTable.DataKey.STATUS_NEW),
        ]
        command = 'CREATE TABLE IF NOT EXISTS {} ({});'.format(TimestampTable.NAME, ', '.join(entries))
        return command


class LogTable(object):

    NAME = 'logs'
    ID_KEY = 'id'
    JOB_ID_KEY ='job_id'

    class DataKey(object):
        TIMESTAMP = 'timestamp'
        LOG_FILE_NAME = 'log_file_name'

    @staticmethod
    def get_creation_command():
        entries = [
            '{} INTEGER PRIMARY KEY AUTOINCREMENT'.format(LogTable.ID_KEY),
            '{} INTEGER NOT NULL'.format(LogTable.JOB_ID_KEY),
            '{} TEXT NOT NULL'.format(LogTable.DataKey.TIMESTAMP),
            '{} TEXT NOT NULL'.format(LogTable.DataKey.LOG_FILE_NAME),
        ]
        command = 'CREATE TABLE IF NOT EXISTS {} ({});'.format(LogTable.NAME, ', '.join(entries))
        return command


class QueueManager(object):
    def __init__(self, queue_directory, make_directories=False):
        self.queue_directory = os.path.abspath(queue_directory)
        self._pid = os.getpid()
        self._timestamp = get_timestamp_string()
        self._prepare_directories(make_directories)
        self._prepare_database()

    def _prepare_directories(self, make_directories):
        if not os.path.isdir(self.queue_directory):
            if make_directories:
                os.makedirs(self.queue_directory)
            else:
                raise Exception(self._directory_error_message(self.queue_directory))
        for folder in [DATABASE_FOLDER, CONFIG_FOLDER, LOG_FOLDER]:
            abs_folder = os.path.join(self.queue_directory, folder)
            if not os.path.isdir(abs_folder):
                if make_directories:
                    os.makedirs(abs_folder)
                else:
                    raise Exception(self._directory_error_message(abs_folder))

    @staticmethod
    def _directory_error_message(path):
        return '[ERROR] {} is not a valid directory path'.format(path)

    def _database_file(self):
        return os.path.join(self.queue_directory, DATABASE_FOLDER, DATABASE_FILE)

    def _prepare_database(self):
        with self._get_database_connection(exclusive=True) as connection:
            self._execute_database_query(connection, JobTable.get_creation_command())
            self._execute_database_query(connection, TimestampTable.get_creation_command())
            self._execute_database_query(connection, LogTable.get_creation_command())
            connection.commit()

    def _get_database_connection(self, exclusive=False):
        isolation_level = 'EXCLUSIVE' if exclusive else None
        connection = sqlite3.connect(self._database_file(), isolation_level=isolation_level)
        if exclusive:
            cursor = connection.cursor()
            cursor.execute('BEGIN EXCLUSIVE')
            connection.commit()
        return connection

    def _execute_database_query(self, connection, query, values=None):
        cursor = connection.cursor()
        try:
            if values is None:
                cursor.execute(query)
            elif type(values) == tuple:
                cursor.execute(query, values)
            elif type(values) == list:
                cursor.executemany(query, values)
            else:
                raise Exception()
        except SQLError as ex:
            raise Exception(ex)
        else:
            return cursor

    def _get_timestamp(self):
        return get_timestamp_string(format='%Y-%m-%d %H:%M:%S.%f')

    def _add_jobs(self, connection, *jobs):
        job_ids = []
        for job in jobs:
            assert isinstance(job, ComputeJob)
            job_id, timestamp = self._insert_job_template(connection)
            config_file_name = self._store_job_config(job.to_dict(), timestamp, job_id)
            self._set_job_data(connection, job_id, JobTable.DataKey.CONFIG_FILE_NAME, config_file_name, timestamp)
            self._set_job_data(connection, job_id, JobTable.DataKey.STATUS, JobStatus.OPEN.value, timestamp)
            self._add_status_change_timestamp(connection, job_id, JobStatus.TBA.value, JobStatus.OPEN.value, timestamp)
            job_ids.append(job_id)
        connection.commit()
        return job_ids

    def add_jobs(self, *jobs):
        with self._get_database_connection(exclusive=True) as connection:
            job_ids = self._add_jobs(connection, *jobs)
            connection.commit()
        return job_ids

    def add_job(self, job):
        return self.add_jobs(job)[0]

    def _insert_job_template(self, connection):
        timestamp = self._get_timestamp()
        fields = ', '.join([JobTable.DataKey.STATUS, JobTable.DataKey.CONFIG_FILE_NAME, JobTable.DataKey.LAST_MODIFIED])
        query = f'INSERT INTO {JobTable.NAME} ({fields}) values (?, ?, ?);'
        values = (JobStatus.TBA.value, 'tba', timestamp)
        cursor = self._execute_database_query(connection, query, values=values)
        return cursor.lastrowid, timestamp

    def _store_job_config(self, config, timestamp, job_id):
        date, time = timestamp.split(' ')
        rel_path = os.path.join(CONFIG_FOLDER, date)
        abs_path = os.path.join(self.queue_directory, rel_path)
        if not os.path.isdir(abs_path):
            os.makedirs(abs_path)
        config_file_name = '.'.join(['job', f'{job_id}', 'json'])
        with open(os.path.join(abs_path, config_file_name), 'a') as config_file:
            json.dump(config, config_file, indent=4)
        return os.path.join(rel_path, config_file_name)

    def _set_job_data(self, connection, job_id, key, value, timestamp):
        query = f'UPDATE {JobTable.NAME} SET {key} = ?, {JobTable.DataKey.LAST_MODIFIED} = ? WHERE {JobTable.ID_KEY} = ?;'
        values = (value, timestamp, job_id)
        self._execute_database_query(connection, query, values=values)

    def _add_status_change_timestamp(self, connection, job_id, old_status, new_status, timestamp):
        fields = ', '.join([
            TimestampTable.JOB_ID_KEY,
            TimestampTable.DataKey.STATUS_OLD,
            TimestampTable.DataKey.STATUS_NEW,
            TimestampTable.DataKey.TIMESTAMP
        ])
        query = f'INSERT INTO {TimestampTable.NAME} ({fields}) VALUES (?, ?, ?, ?);'
        values = (job_id, old_status, new_status, timestamp)
        self._execute_database_query(connection, query, values=values)

    def retrieve_job(self, lock=True):
        with self._get_database_connection(exclusive=True) as connection:
            timestamp = self._get_timestamp()
            cursor = self._filter_table(
                connection,
                JobTable, JobTable.DataKey.STATUS, JobStatus.OPEN.value,
                retrieve_keys=[JobTable.ID_KEY, JobTable.DataKey.CONFIG_FILE_NAME]
            )
            response = cursor.fetchone()
            if response is not None:
                job_id, config_file_name = response
                self._set_job_data(connection, job_id, JobTable.DataKey.STATUS, JobStatus.PENDING.value, timestamp)
                self._add_status_change_timestamp(connection, job_id, JobStatus.OPEN.value, JobStatus.PENDING.value, timestamp)
            connection.commit()
        if response is None:
            return None
        config, job_lock = self._checkout_job_config(config_file_name, lock)
        job = ComputeJob.from_dict(config, id=job_id)
        return (job, job_lock) if lock else job

    def _filter_table(self, connection, table, key, values, retrieve_keys=None):
        if retrieve_keys is None:
            retrieve_string = '*'
        else:
            if not type(retrieve_keys) == list:
                retrieve_keys = [retrieve_keys]
            retrieve_string = ', '.join(retrieve_keys)
            if len(retrieve_keys) > 1:
                retrieve_string = retrieve_string
        if not type(values) == tuple:
            if not type(values) == list:
                values = [values]
            values = tuple(values)
        if len(values) == 1:
            where_clause = f'{key} = ?'
        else:
            qmarks = ', '.join(['?'] * len(values))
            where_clause = f'{key} IN ({qmarks})'
        query = f'SELECT {retrieve_string} FROM {table.NAME} WHERE {where_clause};'
        cursor = self._execute_database_query(connection, query, values)
        return cursor

    def _checkout_job_config(self, config_file_name, lock):
        config_file = open(self._get_config_file_path(config_file_name), 'r')
        if lock:
            fcntl.flock(config_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.load(config_file)
        if not lock:
            config_file.close()
        return config, config_file if lock else config

    def unlock_job(self, job_lock):
        fcntl.flock(job_lock, fcntl.LOCK_UN)

    def _get_config_file_path(self, config_file_name):
        return os.path.join(self.queue_directory, config_file_name)

    def remove_jobs(self, *job_ids, remove_config_files=True, remove_timestamps=True):
        with self._get_database_connection(exclusive=True) as connection:
            self._remove_jobs(connection, *job_ids, remove_config_files=remove_config_files, remove_timestamps=remove_timestamps)
            connection.commit()

    def remove_job(self, job_id, remove_config_files=True, remove_timestamps=True):
        return self.remove_jobs(
            self, job_id,
            remove_config_files=remove_config_files, remove_timestamps=remove_timestamps
        )

    def _remove_jobs(self, connection, *job_ids, remove_config_files=True, remove_timestamps=True):
        config_file_names = []
        if remove_config_files:
            config_file_names = self._filter_table(
                connection,
                JobTable, JobTable.ID_KEY, list(job_ids),
                retrieve_keys=JobTable.DataKey.CONFIG_FILE_NAME
            ).fetchall()
        self._remove_entries(connection, JobTable, JobTable.ID_KEY, list(job_ids))
        if remove_config_files:
            self._remove_config_files(*[cfn[0] for cfn in config_file_names])
        if remove_timestamps:
            self._remove_entries(connection, TimestampTable, TimestampTable.JOB_ID_KEY, list(job_ids))
        return list(job_ids)

    def _remove_entries(self, connection, table, key, values):
        if not type(values) == tuple:
            if not type(values) == list:
                values = [values]
            values = tuple(values)
        if len(values) == 1:
            where_clause = f'{key} = ?'
        else:
            qmarks = ', '.join(['?'] * len(values))
            where_clause = f'{key} IN {qmarks}'
        query = f'DELETE FROM {table.NAME} WHERE {where_clause};'
        self._execute_database_query(connection, query, values=values)

    def _remove_config_files(self, *config_file_names):
        for config_file_name in config_file_names:
            full_path = os.path.join(self.queue_directory, config_file_name)
            os.remove(full_path)

    def remove_dead_jobs(self, remove_config_files=True, remove_timestamps=True):
        with self._get_database_connection(exclusive=True) as connection:
            dead_job_ids = self._find_dead_jobs(connection)
            self._remove_jobs(
                connection, *dead_job_ids,
                remove_config_files=remove_config_files, remove_timestamps=remove_timestamps
            )
        return dead_job_ids

    def _find_dead_jobs(self, connection):
        dead_job_ids = []
        cursor = self._filter_table(
            connection,
            JobTable, JobTable.DataKey.STATUS, [ComputeJob.Status.PENDING.value, ComputeJob.Status.RUNNING.value],
            retrieve_keys=[JobTable.ID_KEY, JobTable.DataKey.CONFIG_FILE_NAME]
        )
        response = cursor.fetchall()
        for job_id, config_file_name in response:
            try:
                file = open(self._get_config_file_path(config_file_name))
            except IOError:
                continue
            else:
                file.close()
                dead_job_ids.append(job_id)
        return dead_job_ids

    def set_job_status(self, job_id, status):
        with self._get_database_connection(exclusive=True) as connection:
            cursor = self._filter_table(connection, JobTable, JobTable.ID_KEY, job_id, retrieve_keys=[JobTable.DataKey.STATUS])
            old_status = cursor.fetchone()[0]
            timestamp = self._get_timestamp()
            self._set_job_data(connection, job_id, JobTable.DataKey.STATUS, status.value, timestamp)
            self._add_status_change_timestamp(connection, job_id, old_status, status.value, timestamp)
            connection.commit()

    def restore_failed_jobs(self):
        with self._get_database_connection(exclusive=True) as connection:
            cursor = self._filter_table(
                connection,
                JobTable, JobTable.DataKey.STATUS, JobStatus.FAILED.value,
                retrieve_keys=[JobTable.ID_KEY, JobTable.DataKey.CONFIG_FILE_NAME]
            )
            response = cursor.fetchall()
            if response is not None:
                jobs = [ComputeJob.from_dict(config) for _, config in response]
                self._add_jobs(connection, *jobs)
            connection.commit()

    def get_log_file(self, job_id):
        with self._get_database_connection(exclusive=True) as connection:
            fields = ', '.join([LogTable.JOB_ID_KEY, LogTable.DataKey.TIMESTAMP, LogTable.DataKey.LOG_FILE_NAME])
            template_query = f'INSERT INTO {LogTable.NAME} ({fields}) VALUES (?, ?, ?)'
            timestamp = self._get_timestamp()
            template_values = (job_id, timestamp, 'tba')
            cursor = self._execute_database_query(connection, template_query, template_values)
            log_id = cursor.lastrowid
            log_file_name = self._create_log_file(job_id, log_id, timestamp)
            update_query = f'UPDATE {LogTable.NAME} SET {LogTable.DataKey.LOG_FILE_NAME} = ? WHERE {LogTable.ID_KEY} = ?;'
            update_values = (log_file_name, log_id)
            self._execute_database_query(connection, update_query, update_values)
            connection.commit()
        return os.path.join(self.queue_directory, log_file_name)

    def _create_log_file(self, job_id, log_id, timestamp):
        date = timestamp.split(' ')[0]
        log_folder = os.path.join(self.queue_directory, LOG_FOLDER, date)
        file_name = f'job.{job_id}.{log_id}.log'
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)
        with open(os.path.join(log_folder, file_name), 'a') as f:
            f.write(f'[INFO] Log file created on {timestamp}')
        return os.path.join(LOG_FOLDER, date, file_name)



def _test():
    # Job creator process
    job = ComputeJob.from_dict(...)
    queue = QueueManager(queue_directory='./test_queue')
    queue.add_job(job)

    # Launcher Process:
    queue = QueueManager(queue_directory='./test_queue')
    job, job_lock = queue.retrieve_job(lock=True)
    queue.set_job_status(job.id, JobStatus.RUNNING)
    response = job.run()
    if response.returncode == 0:
        queue.set_job_status(job.id, JobStatus.FINISHED)
    else:
        queue.set_job_status(job.id, JobStatus.FAILED)
    queue.unlock_job(job_lock)

    # Cleanup:
    queue = QueueManager(queue_directory='./test_queue')
    queue.remove_dead_jobs()
