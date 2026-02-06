.. _olympus:

.. This target is required to prevent the Sphinx build error "Unknown target name: olympus".
.. The custom directive list_config_options imports OLYMPUS to extract real configuration
.. data, which causes Sphinx to look for a target named "olympus". This dummy target
.. satisfies that requirement while allowing the actual OLYMPUS import to work.

Configuration Options
=====================

OLYMPUS provides various configuration options to customize its behavior. These options control everything from numerical precision to debugging features.

How to Use Configuration Options
--------------------------------

OLYMPUS configuration options can be set in several ways:

1. **Environment variables** (set before running your program):

   .. code-block:: bash

      export OLYMPUS_ENABLE_X64=True
      python my_program.py

2. **Runtime configuration** (in your Python code):

   .. code-block:: python

      import olympus
      olympus.config.update("olympus_enable_x64", True)

3. **Command-line flags** (using Abseil):

   .. code-block:: python

      # In your code:
      import olympus
      olympus.config.parse_flags_with_absl()

   .. code-block:: bash

      # When running:
      python my_program.py --olympus_enable_x64=True

Common Configuration Options
----------------------------

Here are some of the most frequently used configuration options:

- ``olympus_enable_x64`` -- Enable 64-bit floating-point precision
- ``olympus_disable_jit`` -- Disable JIT compilation for debugging
- ``olympus_debug_nans`` -- Check for and raise errors on NaNs
- ``olympus_platforms`` -- Control which backends (CPU/GPU/TPU) OLYMPUS will initialize
- ``olympus_numpy_rank_promotion`` -- Control automatic rank promotion behavior
- ``olympus_default_matmul_precision`` -- Set default precision for matrix multiplication operations

.. raw:: html

   <div style="margin-top: 30px;"></div>

All Configuration Options
-------------------------

Below is a complete list of all available OLYMPUS configuration options:

.. list_config_options::
