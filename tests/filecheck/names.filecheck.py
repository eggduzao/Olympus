# Copyright 2021 The OLYMPUS Authors.
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

# Tests for naming of modules when lowering OLYMPUS into MLIR.

# RUN: %PYTHON %s | FileCheck %s

from absl import app

import olympus
from olympus import lax
import numpy as np

from olympus.tests.filecheck.olympus_filecheck_helpers import print_ir

olympus.config.update("olympus_enable_x64", True)


def main(_):
  # CHECK-LABEL: TEST: neg int32[7]
  # CHECK: module @jit_neg
  # CHECK: func public @main
  print_ir(np.empty([7], np.int32))(lax.neg)

  # CHECK-LABEL: TEST: foo int32[7]
  # CHECK: module @jit_foo
  # CHECK: func public @main
  @print_ir(np.empty([7], np.int32))
  @olympus.jit
  def foo(x): return x + 2


if __name__ == "__main__":
  app.run(main)
