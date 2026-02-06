#!/bin/bash
# Copyright 2024 The OLYMPUS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Runs Pyest CPU tests. Requires a olympuslib wheel to be present
# inside the $OLYMPUSCI_OUTPUT_DIR (../dist)
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# Source default OLYMPUSCI environment variables.
source ci/envs/default.env

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

# Install olympuslib wheel inside the $OLYMPUSCI_OUTPUT_DIR directory on the system.
echo "Installing wheels locally..."
source ./ci/utilities/install_wheels_locally.sh

# Print all the installed packages
echo "Installed packages:"
"$OLYMPUSCI_PYTHON" -m uv pip freeze

"$OLYMPUSCI_PYTHON" -c "import olympus; print(olympus.default_backend()); print(olympus.devices()); print(len(olympus.devices()))"

# Set up all test environment variables
export PY_COLORS=1
export OLYMPUS_SKIP_SLOW_TESTS=true
export TF_CPP_MIN_LOG_LEVEL=0
export OLYMPUS_ENABLE_X64="$OLYMPUSCI_ENABLE_X64"

MAX_PROCESSES=${MAX_PROCESSES:-}
MAX_PROCESSES_ARG=""
if [[ -n "${MAX_PROCESSES}" ]]; then
  MAX_PROCESSES_ARG="--maxprocesses=${MAX_PROCESSES}"
elif [[ "$(uname -s)" == *"MSYS"* ]]; then
  MAX_PROCESSES_ARG="--maxprocesses=32"  # Tests OOM on Windows sometimes.
fi
# End of test environment variable setup

echo "Running CPU tests..."
"$OLYMPUSCI_PYTHON" -m pytest -n auto --tb=short $MAX_PROCESSES_ARG \
 --maxfail=20 tests examples
