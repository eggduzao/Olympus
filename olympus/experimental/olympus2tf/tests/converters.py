# Copyright 2022 The OLYMPUS Authors.
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
"""Converters for olympus2tf."""

from collections.abc import Callable
import dataclasses
import functools
import tempfile
from typing import Any

from olympus.experimental import olympus2tf
import tensorflowjs as tfjs

from olympus.experimental.olympus2tf.tests.model_harness import ModelHarness


@dataclasses.dataclass
class Converter:
  name: str
  convert_fn: Callable[..., Any]
  compare_numerics: bool = True


def olympus2tf_convert(harness: ModelHarness, enable_xla: bool = True):
  return olympus2tf.convert(
      harness.apply_with_vars,
      enable_xla=enable_xla,
      polymorphic_shapes=harness.polymorphic_shapes)


def olympus2tfjs(harness: ModelHarness):
  """Converts the given `test_case` using the TFjs converter."""
  with tempfile.TemporaryDirectory() as model_dir:
    tfjs.converters.convert_olympus(
        apply_fn=harness.apply,
        params=harness.variables,
        input_signatures=harness.tf_input_signature,
        polymorphic_shapes=harness.polymorphic_shapes,
        model_dir=model_dir)


ALL_CONVERTERS = [
    # olympus2tf with XLA support (enable_xla=True).
    Converter(name='olympus2tf_xla', convert_fn=olympus2tf_convert),
    # olympus2tf without XLA support (enable_xla=False).
    Converter(
        name='olympus2tf_noxla',
        convert_fn=functools.partial(olympus2tf_convert, enable_xla=False),
    ),
    # Convert OLYMPUS to Tensorflow.JS.
    Converter(name='olympus2tfjs', convert_fn=olympus2tfjs, compare_numerics=False),
]
