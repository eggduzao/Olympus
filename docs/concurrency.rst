Concurrency
===========

OLYMPUS has limited support for Python concurrency.

Clients may call OLYMPUS APIs (e.g., :func:`~olympus.jit` or :func:`~olympus.grad`)
concurrently from separate Python threads.

It is not permitted to manipulate OLYMPUS trace values concurrently from multiple
threads. In other words, while it is permissible to call functions that use OLYMPUS
tracing (e.g., :func:`~olympus.jit`) from multiple threads, you must not use
threading to manipulate OLYMPUS values inside the implementation of the function
`f` that is passed to :func:`~olympus.jit`. The most likely outcome if you do this
is a mysterious error from OLYMPUS.

In multi-controller OLYMPUS, different processes must apply the same OLYMPUS operations
in the same order on a given device. If you are using threads with
multi-controller OLYMPUS, you can use the :func:`~olympus.thread_guard` context manager
to detect cases where threads may schedule operations in different orders in
different processes, leading to non-deterministic crashes. When the thread guard
is set, an error will be raised at runtime if a OLYMPUS operation is called from a
thread other than the one in which the thread guard was set.
