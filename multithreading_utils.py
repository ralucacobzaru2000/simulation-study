import os
import sys
import time
import traceback
from typing import Any, Callable, List, Optional

import psutil
import multiprocessing as m
import threading as t
from pygam.utils import OptimizationError
from scipy.linalg.misc import LinAlgError


class MultithreadingUtils:
    WAIT_TIME: float = 1

    def run_function(
        self,
        func: Callable,
        args_list: List[List[Any]],
        show_prints: bool = False,
        workers: Optional[int] = None,
    ) -> List[Any]:
        manager = m.Manager()

        # number of iterations completed
        process_counter = ProcessCounter(manager)
        iterations_done = ThreadCounter(process_counter)
        error_list = manager.list()
        results = []

        process_errors = t.Event()
        print_done = t.Event()

        scheduler = Scheduler(
            process_errors,
            results,
            func,
            args_list,
            process_counter,
            error_list,
            show_prints,
            workers,
        )
        progress_status = PrintProgress(print_done, iterations_done, len(args_list))

        scheduler.start()
        progress_status.start()

        # check where we are at right now
        while iterations_done.get() < len(args_list):
            if len(error_list) > 0:
                process_errors.set()
                print_done.set()
                print("Errors encountered in child processes:\n")
                for i in range(len(error_list)):
                    print(f"[Error {i + 1}/{len(error_list)}]\n")
                    for error in error_list[i]:
                        print(error)
                print("Stopping main process.")
                sys.exit(1)

            time.sleep(self.WAIT_TIME)

        scheduler.join()

        time.sleep(self.WAIT_TIME)
        print_done.set()
        progress_status.join()
        print("", file=sys.stderr)

        return results


# class to supress print statements
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class ProcessCounter:
    def __init__(self, manager: m.Manager) -> None:
        self.counter = manager.Value("i", 0)
        self._m_lock = manager.Lock()

    def increment(self) -> None:
        with self._m_lock:
            self.counter.value += 1

    def get(self) -> int:
        with self._m_lock:
            return self.counter.value


class ThreadCounter:
    def __init__(self, process_cunter: ProcessCounter) -> None:
        self.counter = process_cunter
        self._t_lock = t.Lock()

    def get(self) -> int:
        with self._t_lock:
            return self.counter.get()


# wrapper function so that we can handle errors and track progress
def func_wrapper(args, func, iterations_done, error_list, show_prints):
    while True:
        try:
            if show_prints:
                result = func(*args)
            else:
                with HiddenPrints():
                    result = func(*args)
            iterations_done.increment()
            return result
            break
        except OptimizationError:
            print("Optimization error in MUtils")
        except LinAlgError:
            print("SVD error")
        except Exception:
            formatted_lines = traceback.format_exc().splitlines()
            error_list.append(formatted_lines)
            sys.exit(1)


class Scheduler(t.Thread):
    WAIT_TIME: float = 0.1
    CPU_MULT: int = 3

    def __init__(
        self,
        should_stop: t.Event,
        results: List,
        func: Callable,
        args_list: List[List[Any]],
        iterations_done: ProcessCounter,
        error_list: List[str],
        show_prints: bool,
        workers: Optional[int],
    ) -> None:
        super().__init__(daemon=True)

        self.iterations_done = iterations_done
        self.error_list = error_list
        self.args_list = args_list
        self.show_prints = show_prints
        self.should_stop = should_stop
        self.results = results
        self.func = func
        self.workers = (
            self.CPU_MULT * psutil.cpu_count(logical=True)
            if workers is None
            else workers
        )

    def run(self) -> None:
        processes = []
        args_left_list = self.args_list.copy()[::-1]

        with m.Pool(self.workers) as pool:
            while len(processes) < len(self.args_list):
                if self.should_stop.is_set():
                    pool.terminate()
                    return

                waiting_processes = sum(
                    1 for process in processes if not process.ready()
                )
                if waiting_processes < self.workers:
                    new_args_len = min(
                        self.workers - waiting_processes, len(args_left_list)
                    )
                    new_args = [args_left_list.pop() for i in range(new_args_len)]
                    for args in new_args:
                        func_args = (
                            args,
                            self.func,
                            self.iterations_done,
                            self.error_list,
                            self.show_prints,
                        )
                        process = pool.apply_async(func_wrapper, args=func_args)
                        processes.append(process)
                time.sleep(self.WAIT_TIME)

            while sum(1 for p in processes if p.ready()) < len(self.args_list):
                if self.should_stop.is_set():
                    pool.terminate()
                    return
                time.sleep(self.WAIT_TIME)

            self.results.extend([process.get() for process in processes])


class PrintProgress(t.Thread):
    TIME_ESTIMATE_REFRESH_RATE: float = 1
    WAIT_TIME: float = 1

    MIN_IN_SECS: int = 60
    HOUR_IN_MINS: int = 60

    def __init__(
        self, print_done: t.Event, iterations_done: ThreadCounter, iterations_total: int
    ) -> None:
        super().__init__(daemon=True)
        self.print_done = print_done
        self.iterations_done = iterations_done
        self.iterations_total = iterations_total
        self.last_fraction = 0
        self.last_time = time.time()
        self.initial_time = self.last_time

    def run(self) -> None:
        while not self.print_done.is_set():
            self.__print_progress()
            time.sleep(self.WAIT_TIME)

    # print current progress
    def __print_progress(self) -> None:
        fraction_done = self.iterations_done.get() / self.iterations_total

        if fraction_done > 0:
            if (
                fraction_done > self.last_fraction
                or time.time() - self.last_time > self.TIME_ESTIMATE_REFRESH_RATE
            ):
                self.last_fraction = fraction_done
                self.last_time = time.time()
            elapsed_time = self.last_time - self.initial_time
            fraction_done = self.last_fraction
            remaining_time = (1.0 / fraction_done) * elapsed_time - elapsed_time
            minutes, sec = divmod(remaining_time, self.MIN_IN_SECS)
            hr, minutes = divmod(minutes, self.HOUR_IN_MINS)

            print(
                f"\rThe task is {100 * fraction_done:.2f}% complete. "
                f"Estimated time remaining is {hr:.0f}:{minutes:02.0f}:{sec:02.0f}",
                file=sys.stderr,
                end="",
            )
        else:
            print(
                f"\rThe task is {100 * fraction_done:.2f}% complete. "
                "Estimated time remaining is unknown.",
                file=sys.stderr,
                end="",
            )
