from __future__ import absolute_import

import datetime
import inspect
import os
import pdb
import shutil
import subprocess

from git import repo
from scipy.misc import toimage

# the existence of this file means the experiment is running.
RUNNING_LOCK = "running.{0}"
SNAPSHOTS_NAME = "snapshots"

class RunningLockFile(object):
    FILENAME_FORMAT = "running.{0}"
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.filename = RunningLockFile.FILENAME_FORMAT.format(os.getpid())

    @property
    def path(self):
        return os.path.join(self.base_path, self.filename)

    def start(self):
        with open(self.path, "a") as f:
            pass

        return self

    def end(self):
        os.remove(self.path)

    def __enter__(self):
        return self.start()

    def __exit__(self, type, value, tb):
        self.end()

class SaveCodeChanges(object):
    DIFF_EXTENSION = os.extsep + "diff"
    
    def __init__(self, repositories):
        self.repositories = repositories

    def __call__(self, path):
        repository_paths = []
        
        for repository in self.repositories:
            if isinstance(repository, str):
                repository_paths.append(repository)
            elif hasattr(repository.__path__, "_path"):
                repository_paths.append(repository.__path__._path[0])
            else:
                repository_paths.append(repository.__path__[0])
            
        for repository_path in repository_paths:
            # create a git diff of the changes against head.
            current_repository = repo.Repo(repository_path)
            repository_changes = current_repository.git.diff()
            diff_name = os.path.basename(repository_path) + SaveCodeChanges.DIFF_EXTENSION
            
            with open(os.path.join(path, diff_name), "w") as diff_file:
                diff_file.write(repository_changes)

class PrettifySessionName(object):
    def __call__(self, run):
        current_name = os.getenv("STY")
        if current_name is None:
            # not inside a screen.
            return

        gpu_number = os.getenv("CUDA_VISIBLE_DEVICES")
        pretty_name = run.experiment_name + "/" + run.run_name  + "@gpu"+ gpu_number
        subprocess.call(["screen", "-X", "title", pretty_name])

class ExperimentRun(object):
    def __init__(self, path, experiment_name, description, startup_tasks=None, cleanup_tasks=None, epitaph=None):
        self.path = path
        self.experiment_name = experiment_name
        self.description = description
        self.epitaph = epitaph
        self.startup_tasks = startup_tasks or []
        self.cleanup_tasks = cleanup_tasks or []
        self.run_name = None
        self.running_lock = None
    
    @property
    def data_path(self):
        return os.path.join(self.path, "data")

    @property
    def checkpoints_path(self):
        return os.path.join(self.path, "checkpoints")

    @property
    def snapshots_path(self):
        return os.path.join(self.path, "snapshots")

    def __call__(self, experiment):
        pass

    def __enter__(self):
        self.run_name = os.path.basename(self.path)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.running_lock = RunningLockFile(self.path)            

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        if not os.path.exists(self.checkpoints_path):            
            os.mkdir(self.checkpoints_path)

        if not os.path.exists(self.snapshots_path):
            os.mkdir(self.snapshots_path)

        self.file = self._create_experiment_file()
        if not (self.file is None):
            self.file.metadata.experiment_name = self.experiment_name
            self.file.metadata.run_name = self.run_name
            self.file.metadata.description = self.description
            self.file.metadata.epitaph = self.epitaph
            self.file.metadata.pid = os.getpid()
            self.file.metadata.devices = os.getenv("CUDA_VISIBLE_DEVICES")            

        self.running_lock.__enter__()    
        print("experiment started in {0}".format(self.path))

        return self

    def end(self):
        self.running_lock.end()

        for cleanup_task in self.cleanup_tasks:        
            try:
                cleanup_task(self)
            except:
                print("encountered error cleaning up")
                
        # here we should do some post-processing e.g. zipping the run.
        print("experiment concluded in {0}".format(self.path))

        if not (self.file is None):
            self.file.close()

        # if the program exited with a legitimate exception, ask
        # if it should be deleted.
        if not (type is None) and issubclass(type, Exception):
            valid_responses = ["n", "y", "N", "Y"]
            while True:
                response = raw_input("\033[91mExperiment ended in fatal exception, should I delete?\033[0m [y/n]")
                if response in valid_responses:
                    break

            if response.upper() == "Y":
                shutil.rmtree(self.run_path)
        
    def __exit__(self, type, value, tb):
        self.end()
