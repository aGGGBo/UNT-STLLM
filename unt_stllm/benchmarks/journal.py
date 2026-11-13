import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ..reporting import save_json


class ExperimentLog:
    def __init__(self, folder, job, mode, stage, number, total):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.start = self.last = time.monotonic()
        task = job.get('task', job.get('target'))
        self.prefix = f'[reproduce][{job["id"]}][seed={job["seed"]}][{job["condition"]}/{stage}/support={.1 if stage in ("independent", "source_shared") else 1.}]'
        self.state = dict(status='running', id=job['id'], task=task, condition=job['condition'], mode=mode,
                          stage=stage, seed=job['seed'], pid=os.getpid(), output=str(self.folder.resolve()),
                          started_at=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
                          exit_code=None, error_tail='')
        self.stream = (self.folder / 'run.log').open('a', encoding='utf8', buffering=1)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        save_json(self.folder / 'status.json', self.state)
        self.emit('start', job=f'{number}/{total}', stage=stage, mode=mode, output=self.folder)
        self.worker = threading.Thread(target=self._heartbeat, daemon=True)
        self.worker.start()

    def emit(self, event, **values):
        with self.lock:
            self.last = time.monotonic()
            message = ' '.join(f'{key}={value}' for key, value in values.items())
            line = f'{self.prefix} {datetime.now().astimezone().strftime("%H:%M:%S")} | {event:<10} | {message}'
            print(line, flush=True)
            self.stream.write(line + '\n')

    def _heartbeat(self):
        while not self.stop.wait(10):
            if time.monotonic() - self.last >= 60:
                self.emit('heartbeat', status='running', elapsed=f'{time.monotonic()-self.start:.1f}s')

    def finish(self, success, error=''):
        self.state.update(status='success' if success else 'failed', exit_code=0 if success else 1,
                          elapsed_seconds=time.monotonic()-self.start,
                          ended_at=datetime.now(timezone.utc).isoformat(), error_tail=error[-12000:])
        save_json(self.folder / 'status.json', self.state)
        self.emit('finish', status=self.state['status'], exit_code=self.state['exit_code'],
                  elapsed=f'{self.state["elapsed_seconds"]:.1f}s', log=self.folder / 'run.log')

    def close(self):
        self.stop.set()
        self.worker.join(timeout=1)
        self.stream.close()


class CampaignLock:
    def __init__(self, root):
        self.path = Path(root) / 'campaign.lock'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open('a+b')
        if os.name == 'nt':
            import msvcrt
            self.stream.seek(0)
            if not self.stream.read(1):
                self.stream.write(b'0')
                self.stream.flush()
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, *unused):
        if self.stream is not None:
            self.stream.close()
