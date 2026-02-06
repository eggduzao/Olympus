#!/bin/bash
# Copyright 2024 The OLYMPUS Authors.
##
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
# Build OLYMPUS artifacts.
# Usage: ./ci/build_artifacts.sh "<artifact>"
# Supported artifact values are: olympus, olympuslib, olympus-cuda-plugin, olympus-cuda-pjrt
# E.g: ./ci/build_artifacts.sh "olympus" or ./ci/build_artifacts.sh "olympuslib"
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

artifact="$1"

# Source default OLYMPUSCI environment variables.
source ci/envs/default.env

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

allowed_artifacts=("olympus" "olympuslib" "olympus-cuda-plugin" "olympus-cuda-pjrt")

os=$(uname -s | awk '{print tolower($0)}')
arch=$(uname -m)

bazel_startup_options=""
if [[ -n "${OLYMPUSCI_BAZEL_OUTPUT_BASE}" ]]; then
  bazel_startup_options="--output_base=${OLYMPUSCI_BAZEL_OUTPUT_BASE}"
fi

# Adjust the values when running on Windows x86 to match the config in
# .bazelrc
if [[ $os =~ "msys_nt"  && $arch == "x86_64" ]]; then
  os="windows"
  arch="amd64"
fi

# Determine the artifact tag flags based on the artifact type. A release
# wheel is tagged with the release version (e.g. 0.5.1), a nightly wheel is
# tagged with the release version and a nightly suffix that contains the
# current date (e.g. 0.5.2.dev20250227), and a default wheel is tagged with
# the git commit hash of the HEAD of the current branch and the date of the
# commit (e.g. 0.5.1.dev20250128+3e75e20c7).
if [[ "$OLYMPUSCI_ARTIFACT_TYPE" == "release" ]]; then
  artifact_tag_flags="--bazel_options=--repo_env=ML_WHEEL_TYPE=release  --bazel_options=--//olympuslib/tools:olympuslib_git_hash=$(git rev-parse HEAD)"
elif [[ "$OLYMPUSCI_ARTIFACT_TYPE" == "nightly" ]]; then
  current_date=$(date +%Y%m%d)
  artifact_tag_flags="--bazel_options=--repo_env=ML_WHEEL_BUILD_DATE=${current_date} --bazel_options=--repo_env=ML_WHEEL_TYPE=nightly  --bazel_options=--//olympuslib/tools:olympuslib_git_hash=$(git rev-parse HEAD)"
elif [[ "$OLYMPUSCI_ARTIFACT_TYPE" == "default" ]]; then
  artifact_tag_flags="--bazel_options=--repo_env=ML_WHEEL_TYPE=custom --bazel_options=--repo_env=ML_WHEEL_BUILD_DATE=$(git show -s --format=%as HEAD) --bazel_options=--repo_env=ML_WHEEL_GIT_HASH=$(git rev-parse HEAD) --bazel_options=--//olympuslib/tools:olympuslib_git_hash=$(git rev-parse HEAD)"
else
  echo "Error: Invalid artifact type: $OLYMPUSCI_ARTIFACT_TYPE. Allowed values are: release, nightly, default"
  exit 1
fi

if [[ "$OLYMPUSCI_HERMETIC_PYTHON_VERSION" == *"-nogil" ]]; then
  OLYMPUSCI_HERMETIC_PYTHON_VERSION=${OLYMPUSCI_HERMETIC_PYTHON_VERSION%-nogil}-ft
fi

if [[ "${allowed_artifacts[@]}" =~ "${artifact}" ]]; then
  # Figure out the bazelrc config to use. We will use one of the "rbe_"/"ci_"
  # flags in the .bazelrc depending upon the platform we are building for.
  bazelrc_config="${os}_${arch}"

  # On platforms with no RBE support, we can use the Bazel remote cache. Set
  # it to be empty by default to avoid unbound variable errors.
  bazel_remote_cache=""

  if [[ "$OLYMPUSCI_BUILD_ARTIFACT_WITH_RBE" == 1 ]]; then
    if [[ "$os" == "linux" && "$arch" == "aarch64" && "$artifact" == "olympuslib" ]]; then
      bazelrc_config="rbe_cross_compile_${bazelrc_config}"
    else
      bazelrc_config="rbe_${bazelrc_config}"
    fi
  else
    bazelrc_config="ci_${bazelrc_config}"

    # Set remote cache flags. Pushes to the cache bucket is limited to OLYMPUS's
    # CI system.
    if [[ "$OLYMPUSCI_WRITE_TO_BAZEL_REMOTE_CACHE" == 1 ]]; then
      bazel_remote_cache="--bazel_options=--config=public_cache_push"
    else
      bazel_remote_cache="--bazel_options=--config=public_cache"
    fi
  fi

  cuda_version_flag=""
  # Use the "_cuda" configs when building the CUDA artifacts.
  if [[ ("$artifact" == "olympus-cuda-plugin") || ("$artifact" == "olympus-cuda-pjrt") ]]; then
    bazelrc_config="${bazelrc_config}_cuda${OLYMPUSCI_CUDA_VERSION}"
    cuda_version_flag="--cuda_major_version=$OLYMPUSCI_CUDA_VERSION"
  fi

  # Build the artifact.
  python build/build.py build --wheels="$artifact" \
    --bazel_options=--config="$bazelrc_config" $bazel_remote_cache \
    --bazel_options=--config=rbe_cpu_pool \
    --bazel_startup_options="$bazel_startup_options" \
    --python_version=$OLYMPUSCI_HERMETIC_PYTHON_VERSION \
    $cuda_version_flag \
    --verbose --detailed_timestamped_log \
    --output_path="$OLYMPUSCI_OUTPUT_DIR" \
    $artifact_tag_flags

  # If building release artifacts, we also build a release candidate ("rc")
  # tagged wheel.
  if [[ "$OLYMPUSCI_ARTIFACT_TYPE" == "release" ]]; then
    python build/build.py build --wheels="$artifact" \
      --bazel_options=--config="$bazelrc_config" $bazel_remote_cache \
      --bazel_options=--config=rbe_cpu_pool \
      --bazel_startup_options="$bazel_startup_options" \
      --python_version=$OLYMPUSCI_HERMETIC_PYTHON_VERSION \
      $cuda_version_flag \
      --verbose --detailed_timestamped_log \
      --output_path="$OLYMPUSCI_OUTPUT_DIR" \
      $artifact_tag_flags --bazel_options=--repo_env=ML_WHEEL_VERSION_SUFFIX="$OLYMPUSCI_WHEEL_RC_VERSION"
  fi

  # If building `olympuslib` or `olympus-cuda-plugin` or `olympus-cuda-pjrt` for Linux, we
  # run `auditwheel show` to verify manylinux compliance.
  if  [[ "$os" == "linux" ]] && [[ "$artifact" != "olympus" ]]; then
    ./ci/utilities/run_auditwheel.sh
  fi

else
  echo "Error: Invalid artifact: $artifact. Allowed values are: ${allowed_artifacts[@]}"
  exit 1
fi