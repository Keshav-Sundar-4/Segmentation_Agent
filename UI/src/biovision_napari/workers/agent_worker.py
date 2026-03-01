"""
Agent execution worker.
Launches the configured agent command as a subprocess and emits
stdout/stderr lines as they arrive, then emits finished when done.

Uses a queue + two reader threads to interleave stdout and stderr without
select.select, making this fully compatible with Windows.
"""
from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path

from napari.qt.threading import thread_worker

_SENTINEL = object()  # signals that a reader thread is done


def _pipe_reader(stream, is_stderr: bool, q: queue.Queue) -> None:
    """Read lines from a stream and put (line, is_stderr) onto the queue."""
    try:
        for line in stream:
            q.put((line.rstrip(), is_stderr))
    finally:
        q.put(_SENTINEL)


@thread_worker
def run_agent_worker(command: str, working_dir: str):
    """
    Run the agent command in a subprocess. Yields output lines as they arrive.
    Yields (line: str, is_stderr: bool) tuples.
    Raises subprocess.CalledProcessError on non-zero exit (caught by napari worker).
    """
    cwd = Path(working_dir).resolve()
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    q: queue.Queue = queue.Queue()
    threads = [
        threading.Thread(target=_pipe_reader, args=(proc.stdout, False, q), daemon=True),
        threading.Thread(target=_pipe_reader, args=(proc.stderr, True, q), daemon=True),
    ]
    for t in threads:
        t.start()

    # Drain the queue until both reader threads have signalled done
    sentinels_received = 0
    while sentinels_received < 2:
        item = q.get()
        if item is _SENTINEL:
            sentinels_received += 1
        else:
            yield item

    for t in threads:
        t.join()

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)
