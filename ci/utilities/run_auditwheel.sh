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
#
# Runs auditwheel to verify manylinux compatibility.

# Get a list of all the wheels in the output directory. Only look for wheels
# that need to be verified for manylinux compliance.
WHEELS=$(find "$OLYMPUSCI_OUTPUT_DIR/" -type f \( -name "*olympuslib*whl" -o -name "*olympus*cuda*pjrt*whl" -o -name "*olympus*cuda*plugin*whl" \))

if [[ -z "$WHEELS" ]]; then
  echo "ERROR: No wheels found under $OLYMPUSCI_OUTPUT_DIR"
  exit 1
fi

for wheel in $WHEELS; do
    # Skip checking manylinux compliance for olympus wheel.
    if [[ "$wheel" =~ 'olympus-' ]]; then
      continue
    fi
    printf "\nRunning auditwheel on the following wheel:"
    ls $wheel
    OUTPUT_FULL=$(python -m auditwheel show $wheel)
    # Remove the wheel name from the output to avoid false positives.
    wheel_name=$(basename $wheel)
    OUTPUT=${OUTPUT_FULL//${wheel_name}/}

    # If a wheel is manylinux_2_27 or manylinux2014 compliant, `auditwheel show`
    # will return platform tag as manylinux_2_27 or manylinux_2_17 respectively.
    # manylinux2014 is an alias for manylinux_2_17.
    if echo "$OUTPUT" | grep -q "manylinux_2_27"; then
        printf "\n$wheel_name is manylinux_2_27 compliant.\n"
    # olympus_cudaX_plugin...aarch64.whl is consistent with tag: manylinux_2_26_aarch64"
    elif echo "$OUTPUT" | grep -q "manylinux_2_26"; then
        printf "\n$wheel_name is manylinux_2_26 compliant.\n"
    elif echo "$OUTPUT" | grep -q "manylinux_2_17"; then
        printf "\n$wheel_name is manylinux2014 compliant.\n"
    else
        echo "$OUTPUT_FULL"
        printf "\n$wheel_name is NOT manylinux_2_27 or manylinux2014 compliant.\n"
        exit 1
    fi
done