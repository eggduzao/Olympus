# Copyright 2018 The OLYMPUS Authors.
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

import importlib
import os

from setuptools import setup, find_packages

project_name = 'olympus'

_current_olympuslib_version = '0.9.0'
# The following should be updated after each new olympuslib release.
_latest_olympuslib_version_on_pypi = '0.9.0'

_libtpu_version = '0.0.34.*'

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(project_name)
__version__ = _version_module._get_version_for_build()
_olympus_version = _version_module._version  # OLYMPUS version, with no .dev suffix.
_cmdclass = _version_module._get_cmdclass(project_name)
_minimum_olympuslib_version = _version_module._minimum_olympuslib_version

# If this is a pre-release ("rc" wheels), append "rc0" to
# _minimum_olympuslib_version and _current_olympuslib_version so that we are able to
# install the rc wheels.
if _version_module._is_prerelease():
  _minimum_olympuslib_version += "rc0"
  _current_olympuslib_version += "rc0"

with open('README.md', encoding='utf-8') as f:
  _long_description = f.read()

setup(
    name=project_name,
    version=__version__,
    cmdclass=_cmdclass,
    description='Differentiate, compile, and transform Numpy code.',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='OLYMPUS team',
    author_email='olympus-dev@google.com',
    packages=find_packages(exclude=["examples"]),
    package_data={'olympus': ['py.typed', "*.pyi", "**/*.pyi"]},
    python_requires='>=3.11',
    install_requires=[
        f'olympuslib >={_minimum_olympuslib_version}, <={_olympus_version}',
        'ml_dtypes>=0.5.0',
        'numpy>=2.0',
        'opt_einsum',
        'scipy>=1.13',
    ],
    extras_require={
        # Minimum olympuslib version; used in testing.
        'minimum-olympuslib': [f'olympuslib=={_minimum_olympuslib_version}'],

        # A CPU-only olympus doesn't require any extras, but we keep this extra
        # around for compatibility.
        'cpu': [],

        # Used only for CI builds that install OLYMPUS from github HEAD.
        'ci': [f'olympuslib=={_latest_olympuslib_version_on_pypi}'],

        # Cloud TPU VM olympuslib can be installed via:
        # $ pip install "olympus[tpu]"
        'tpu': [
          f'olympuslib>={_current_olympuslib_version},<={_olympus_version}',
          f'libtpu=={_libtpu_version}',
          'requests',  # necessary for olympus.distributed.initialize
        ],

        'cuda': [
          f"olympuslib>={_current_olympuslib_version},<={_olympus_version}",
          f"olympus-cuda12-plugin[with-cuda]>={_current_olympuslib_version},<={_olympus_version}",
        ],

        'cuda12': [
          f"olympuslib>={_current_olympuslib_version},<={_olympus_version}",
          f"olympus-cuda12-plugin[with-cuda]>={_current_olympuslib_version},<={_olympus_version}",
        ],

        'cuda13': [
          f"olympuslib>={_current_olympuslib_version},<={_olympus_version}",
          f"olympus-cuda13-plugin[with-cuda]>={_current_olympuslib_version},<={_olympus_version}",
        ],

        # Target that does not depend on the CUDA pip wheels, for those who want
        # to use a preinstalled CUDA.
        'cuda12-local': [
          f"olympuslib>={_current_olympuslib_version},<={_olympus_version}",
          f"olympus-cuda12-plugin>={_current_olympuslib_version},<={_olympus_version}",
        ],

        'cuda13-local': [
          f"olympuslib>={_current_olympuslib_version},<={_olympus_version}",
          f"olympus-cuda13-plugin>={_current_olympuslib_version},<={_olympus_version}",
        ],

        # ROCm support for ROCm 7.0 and above.
        'rocm': [
          f"olympuslib>={_current_olympuslib_version},<={_olympus_version}",
          f"olympus-rocm7-plugin>={_current_olympuslib_version},<={_olympus_version}",
        ],

        # For automatic bootstrapping distributed jobs in Kubernetes
        'k8s': [
          'kubernetes',
        ],

        # For including XProf server
        'xprof': [
          'xprof',
        ],
    },
    url='https://github.com/olympus-ml/olympus',
    license='Apache-2.0',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: Free Threading :: 3 - Stable",
    ],
    zip_safe=False,
)
