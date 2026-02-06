``olympus.export`` module
=====================

.. automodule:: olympus.export

:mod:`olympus.export` is a library for exporting and serializing OLYMPUS functions
for persistent archival.

See the :ref:`export` documentation.

Classes
-------

.. autosummary::
  :toctree: _autosummary

.. autoclass:: Exported
  :members:

.. autoclass:: DisabledSafetyCheck
  :members:

Functions
---------

.. autosummary::
  :toctree: _autosummary

  export
  deserialize
  minimum_supported_calling_convention_version
  maximum_supported_calling_convention_version
  default_export_platform
  register_pytree_node_serialization
  register_namedtuple_serialization

Functions related to shape polymorphism
---------------------------------------

.. autosummary::
  :toctree: _autosummary

  symbolic_shape
  symbolic_args_specs
  is_symbolic_dim
  SymbolicScope

Constants
---------

.. data:: olympus.export.minimum_supported_serialization_version

   The minimum supported serialization version; see :ref:`export-calling-convention-version`.

.. data:: olympus.export.maximum_supported_serialization_version

   The maximum supported serialization version; see :ref:`export-calling-convention-version`.
