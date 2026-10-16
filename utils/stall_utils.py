"""
Diagnosis of stalls: while an event (e.g., a pruning round) takes longer than a threshold, a sampler thread records the Python stack of the
main thread after every further threshold; an event that stalled writes its samples to a file and reports what the process did meanwhile:
CPU time and context switches of the main thread, page faults of the process, and CPU throttling of its cgroup. The sampler needs the GIL, so
a sample taken much later than scheduled means that the main thread held the GIL meanwhile (Python code or a C extension that keeps it).
"""
import os
import resource
import sys
import threading
import traceback
from time import perf_counter, sleep


def _cgroup_cpu_stat_path():
    """
    The cpu.stat file of this process's cgroup (v2, or the cpu controller of v1), or None.
    """
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as file:
            for line in file:
                _, controllers, path = line.rstrip("\n").split(":", 2)
                if controllers == "":
                    candidate = f"/sys/fs/cgroup{path}/cpu.stat"
                elif "cpu" in controllers.split(","):
                    candidate = f"/sys/fs/cgroup/{controllers}{path}/cpu.stat"
                else:
                    continue
                if os.path.exists(candidate):
                    return candidate
    except OSError:
        pass
    return None


class StallTracer:
    def __init__(self, threshold, path):
        self.threshold = threshold
        self.path = path
        self.file = open(path, "w", encoding="utf-8")
        self.cgroup_cpu_stat = _cgroup_cpu_stat_path()
        self.stalls = 0
        self.main_thread = threading.get_ident()
        self.lock = threading.Lock()
        self.armed = threading.Event()
        self.event_id, self.start, self.samples, self.closed = 0, None, [], False
        self.sampler = threading.Thread(target=self._sample, daemon=True)
        self.sampler.start()

    def _sample(self):
        while True:
            self.armed.wait()
            with self.lock:
                if self.closed:
                    return
                event_id, start = self.event_id, self.start[0]
            due = start + self.threshold
            while True:
                sleep(max(0.0, due - perf_counter()))
                with self.lock:
                    if self.event_id != event_id or not self.armed.is_set():
                        break
                    frame = sys._current_frames().get(self.main_thread)
                    self.samples.append((due - start, perf_counter() - start, traceback.extract_stack(frame) if frame is not None else []))
                due += self.threshold

    def _throttling(self):
        """
        Number of throttled periods and throttled seconds of the cgroup, or None.
        """
        if self.cgroup_cpu_stat is None:
            return None
        with open(self.cgroup_cpu_stat, encoding="utf-8") as file:
            stat = dict(line.split() for line in file)
        seconds = int(stat["throttled_usec"]) * 1e-6 if "throttled_usec" in stat else int(stat.get("throttled_time", 0)) * 1e-9
        return int(stat.get("nr_throttled", 0)), seconds

    def begin(self):
        start = (perf_counter(), resource.getrusage(resource.RUSAGE_THREAD), resource.getrusage(resource.RUSAGE_SELF), self._throttling())
        with self.lock:
            self.event_id += 1
            self.start, self.samples = start, []
            self.armed.set()

    def end(self, event):
        """
        Returns a description of the event if it stalled (its stack samples then go to the file), else None.
        """
        with self.lock:
            self.armed.clear()
            samples = self.samples
        wall = perf_counter() - self.start[0]
        if wall <= self.threshold:
            return None
        thread, process, throttling = resource.getrusage(resource.RUSAGE_THREAD), resource.getrusage(resource.RUSAGE_SELF), self._throttling()
        _, thread_start, process_start, throttling_start = self.start
        cpu = thread.ru_utime + thread.ru_stime - thread_start.ru_utime - thread_start.ru_stime
        message = (f"Stall: {event} took {wall:.3f} s | main thread: {cpu:.3f} s CPU, {thread.ru_nvcsw - thread_start.ru_nvcsw:d} voluntary / "
                   f"{thread.ru_nivcsw - thread_start.ru_nivcsw:d} involuntary context switches | process: "
                   f"{process.ru_majflt - process_start.ru_majflt:d} major / {process.ru_minflt - process_start.ru_minflt:d} minor page faults")
        if throttling is not None and throttling_start is not None:
            message += (f" | cgroup: throttled {throttling[0] - throttling_start[0]:d} times for "
                        f"{throttling[1] - throttling_start[1]:.3f} s")
        self.file.write(f"=== {message}\n")
        for scheduled, taken, stack in samples:
            self.file.write(f"--- sample due at +{scheduled:.3f} s, taken at +{taken:.3f} s:\n" + "".join(traceback.format_list(stack)))
        self.file.write("\n")
        self.file.flush()
        self.stalls += 1
        return message

    def close(self):
        """
        Stops the sampler and closes the file, which is removed if nothing stalled.
        """
        with self.lock:
            self.closed = True
            self.armed.set()
        self.sampler.join()
        self.file.close()
        if self.stalls == 0:
            os.remove(self.path)
