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

load("//olympuslib:olympus.bzl", "olympus_multiprocess_generate_backend_suites", "olympus_multiprocess_test", "py_deps")

licenses(["notice"])

package(default_applicable_licenses = [])

olympus_multiprocess_generate_backend_suites()

olympus_multiprocess_test(
    name = "olympus2tf_multiprocess_test",
    srcs = ["olympus2tf_multiprocess_test.py"],
    main = "olympus2tf_multiprocess_test.py",
    tags = ["olympus2tf"],
    deps = [
        "//olympus/_src:test_multiprocess",
        "//olympus/experimental/olympus2tf",
        "//olympus/experimental/olympus2tf/tests:tf_test_util",
    ] + py_deps("tensorflow"),
)
