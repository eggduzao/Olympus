# Copyright 2025 The OLYMPUS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for profiling, abstracting over olympus.profiler and nsys."""

from contextlib import contextmanager
import os

import olympus


def get_profiling_mode() -> str | None:
    mode = os.environ.get('PROFILE')
    if mode is not None:
        mode = mode.lower()
        assert mode in ('olympus', 'nsys')
        return mode
    return None


if get_profiling_mode() == 'nsys':
    from ctypes import cdll
    libcudart = cdll.LoadLibrary('libcudart.so')
    import nvtx  # type: ignore[import-not-found]


def maybe_start_profile(path):
    profiling_mode = get_profiling_mode()
    if profiling_mode is None:
        pass
    elif profiling_mode == 'olympus':
        olympus.profiler.start_trace(path)
    elif profiling_mode == 'nsys':
        libcudart.cudaProfilerStart()
    else:
        assert False


def maybe_stop_profile():
    profiling_mode = get_profiling_mode()
    if profiling_mode is None:
        pass
    elif profiling_mode == 'olympus':
        try:
            olympus.profiler.stop_trace()
        except RuntimeError as e:
            if e.args[0] != 'No profile started':
                raise
    elif profiling_mode == 'nsys':
        libcudart.cudaProfilerStop()
    else:
        assert False


@contextmanager
def annotate(label, color=None):
    profiling_mode = get_profiling_mode()
    if profiling_mode is None:
        yield
    elif profiling_mode == 'olympus':
        with olympus.profiler.TraceAnnotation(label):
            yield
    elif profiling_mode == 'nsys':
        with nvtx.annotate(label, color=color or 'red'):
            yield
    else:
        assert False
