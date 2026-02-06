``olympus.ops`` module
==================

.. currentmodule:: olympus.ops

.. automodule:: olympus.ops

.. _syntactic-sugar-for-ops:

The functions ``olympus.ops.index_update``, ``olympus.ops.index_add``, etc., which were
deprecated in OLYMPUS 0.2.22, have been removed. Please use the
:attr:`olympus.numpy.ndarray.at` property on OLYMPUS arrays instead.

Segment reduction operators
---------------------------

.. autosummary::
  :toctree: _autosummary

    segment_max
    segment_min
    segment_prod
    segment_sum
