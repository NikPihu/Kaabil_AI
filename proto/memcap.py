"""Hard memory (and optional CPU) cap for a group of processes on Windows.

A Windows Job Object with JOB_OBJECT_LIMIT_JOB_MEMORY limits the *committed*
memory of every process in the job together.  When the group asks for more,
the allocation fails inside the process (MemoryError / bad_alloc / crash) -
nothing is swapped silently, so "does not fit" shows up as a real failure.

    job = CapJob(limit_mb=2048, cpu_percent=None)
    p = subprocess.Popen(...); job.add(p)
    job.peak_mb()      # peak committed memory of the whole job so far
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]


class EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class CPU_RATE(ctypes.Structure):
    _fields_ = [("ControlFlags", wt.DWORD), ("CpuRate", wt.DWORD)]


JobObjectExtendedLimitInformation = 9
JobObjectCpuRateControlInformation = 15
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x1
JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x4

k32.CreateJobObjectW.restype = wt.HANDLE
k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
k32.QueryInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p]
k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
k32.CloseHandle.argtypes = [wt.HANDLE]


def _check(ok, what):
    if not ok:
        raise OSError(ctypes.get_last_error(), f"{what} failed")


class CapJob:
    def __init__(self, limit_mb: int, cpu_percent: float | None = None):
        self.limit_mb = limit_mb
        self.h = k32.CreateJobObjectW(None, None)
        _check(self.h, "CreateJobObject")
        info = EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        info.JobMemoryLimit = int(limit_mb) * 1024 * 1024
        _check(k32.SetInformationJobObject(self.h, JobObjectExtendedLimitInformation,
                                           ctypes.byref(info), ctypes.sizeof(info)), "SetInformationJobObject")
        if cpu_percent:
            rate = CPU_RATE(JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
                            int(cpu_percent * 100))
            _check(k32.SetInformationJobObject(self.h, JobObjectCpuRateControlInformation,
                                               ctypes.byref(rate), ctypes.sizeof(rate)), "CPU rate")

    def add(self, popen) -> None:
        _check(k32.AssignProcessToJobObject(self.h, wt.HANDLE(int(popen._handle))), "AssignProcessToJobObject")

    def _info(self) -> EXTENDED_LIMIT:
        info = EXTENDED_LIMIT()
        k32.QueryInformationJobObject(self.h, JobObjectExtendedLimitInformation,
                                      ctypes.byref(info), ctypes.sizeof(info), None)
        return info

    def peak_mb(self) -> int:
        return round(self._info().PeakJobMemoryUsed / 1048576)

    def peak_process_mb(self) -> int:
        return round(self._info().PeakProcessMemoryUsed / 1048576)

    def close(self) -> None:
        if self.h:
            k32.CloseHandle(self.h)      # KILL_ON_JOB_CLOSE ends every process in it
            self.h = None

    def __del__(self):
        self.close()
