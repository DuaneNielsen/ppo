import logging
import multiprocessing
import sched
import shutil
import struct
from copy import copy
from pathlib import Path

import tensorboardX

from services.server import Server
from messages import EpisodeMessage, StartMessage
from policy_db import PolicyDB


class TensorBoardCleaner(multiprocessing.Process):
    def __init__(self, db_host ='localhost', db_port=5432, db_name='policy_db', db_user='policy_user', db_password='password',
                 clean_frequency_seconds=4):
        super().__init__()
        self.schedule = sched.scheduler()
        self.clean_frequency_seconds = clean_frequency_seconds
        self.db = PolicyDB(db_host=db_host, db_port=db_port, db_name=db_name, db_user=db_user, db_password=db_password)

    def run(self):
        self.schedule.enter(0, 0, self.clean)
        self.schedule.run()

    def clean(self):
        logging.debug('cleaning')
        runs = self.db.runs()
        files = list(Path('runs').glob('*/events*.*'))
        rundirs = {}

        # get a list of run directories
        for file in files:
            if file.parent.stem not in rundirs:
                rundirs[file.parent.stem] = file.parent

        # dont delete the ones in the policy database
        dirs_to_delete = copy(rundirs)
        for run in runs:
            if run in rundirs:
                del dirs_to_delete[run]

        # cleanup the run directory
        for parent, file in dirs_to_delete.items():
            logging.debug(file.parent, file.name, file.stat().st_size)
            try:
                shutil.rmtree(str(file))
            except:
                logging.error(f"OS didn't let us delete {str(file.parent)}")

        self.schedule.enter(self.clean_frequency_seconds, 0, self.clean)


class TensorBoardStepWriter:
    def __init__(self, rundir):
        self.filepath = Path(rundir) / 'global_step'
        self.f = self.filepath.open('rb+')
        if not self.filepath.exists():
            self.write(0)

    def write(self, tb_step):
        try:
            buffer = struct.pack('i', tb_step)
            self.f.write(buffer)
            self.f.flush()
        except IOError:
            raise

    def read(self):
        buffer = self.f.read(4)
        return struct.unpack('i', buffer)[0]

    def increment(self):
        try:
            self.f.seek(0)
            buffer = self.f.read(4)
            value = struct.unpack('i', buffer)[0]
            self.f.seek(0)
            save_value = value + 1
            buffer = struct.pack('i', save_value)
            self.f.write(buffer)
            self.f.truncate()
            return value
        except:
            raise

    def __del__(self):
        self.f.close()


class TensorBoardListener(Server):
    def __init__(self, redis_host, redis_port, redis_db, redis_password,
        db_host ='localhost', db_port=5432, db_name='policy_db', db_user='policy_user', db_password='password',
                 clean_frequency_seconds=4,
    ):
        super().__init__(redis_host, redis_port, redis_db, redis_password)

        self.handler.register(EpisodeMessage, self.episode)
        self.handler.register(StartMessage, self.start)
        self.tb_step = 0
        self.cleaner = clean_frequency_seconds
        self.cleaner_process = TensorBoardCleaner(db_host=db_host, db_port=db_port, db_name=db_name, db_user=db_user,
                                                  db_password=db_password,
                                                  clean_frequency_seconds=clean_frequency_seconds)

        self.db = PolicyDB(db_host=db_host, db_port=db_port, db_name=db_name, db_user=db_user, db_password=db_password)
        run = self.db.get_latest()
        rundir = 'runs/' + run.run
        self.tb = tensorboardX.SummaryWriter(rundir)
        self.tb_step = TensorBoardStepWriter(rundir)

        self.cleaner_process.start()

    # todo need to handle resuming
    def start(self, msg):
        logging.info('Starting run ' + msg.config.run_id)
        rundir = 'runs/' + msg.config.run_id
        self.tb_step = TensorBoardStepWriter(rundir)
        self.tb = tensorboardX.SummaryWriter(rundir)

    def episode(self, msg):
        tb_step = self.tb_step.increment()
        self.tb.add_scalar('reward', msg.total_reward, tb_step)
        self.tb.add_scalar('epi_len', msg.steps, tb_step)