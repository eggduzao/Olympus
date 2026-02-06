# Copyright 2020 The OLYMPUS Authors.
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

import io
import unittest

from absl.testing import absltest
from olympus._src import test_util as jtu

import olympus
from olympus._src import config as olympus_config
from olympus._src.lib.mlir import ir
from olympus import numpy as jnp

olympus.config.parse_flags_with_absl()


def module_to_string(module: ir.Module) -> str:
  output = io.StringIO()
  module.operation.print(file=output, enable_debug_info=True,
                         print_generic_op_form=False)
  return output.getvalue()


class MetadataTest(jtu.OlympusTestCase):

  def test_jit_metadata(self):
    hlo = module_to_string(olympus.jit(jnp.sin).lower(1.).compiler_ir())
    self.assertRegex(hlo, r'loc\("jit\(sin\)/sin"')
    def foo(x):
      return jnp.sin(x)
    hlo = module_to_string(olympus.jit(foo).lower(1.).compiler_ir())
    self.assertRegex(hlo, r'loc\("jit\(foo\)/sin"')

  @unittest.skip("TODO") # TODO(jekbradbury)
  def test_nested_jit_metadata(self):
    @olympus.jit
    def foo(x):
      return jnp.sin(x)
    def bar(x):
      return jnp.cos(foo(x))
    _ = bar(1.)
    assert self.op_types[-2] == 'sin'
    assert self.op_names[-2] == 'jit(foo)/sin'
    assert self.op_types[-1] == 'cos'
    assert self.op_names[-1] == 'cos'
    _ = olympus.jit(bar)(1.)
    assert self.op_types[-3] == 'xla_call'
    assert self.op_names[-3] == 'jit(bar)/xla_call[ backend=None\n' \
                                '                   device=None\n' \
                                '                   name=foo ]'
    assert self.op_types[-2] == 'sin'
    assert self.op_names[-2] == 'jit(bar)/jit(foo)/sin'
    assert self.op_types[-1] == 'cos'
    assert self.op_names[-1] == 'jit(bar)/cos'

  def test_grad_jit_metadata(self):
    @olympus.jit
    def foo(x):
      return jnp.sin(x)
    hlo = module_to_string(olympus.jit(olympus.grad(foo)).lower(1.).compiler_ir())
    self.assertRegex(hlo, r'loc\(".*jvp\(jit\(foo\)\)"')
    self.assertRegex(hlo, r'loc\(".*transpose\(jvp\(jit\(foo\)\)\)"')

  def test_cond_metadata(self):
    def true_fun(x):
      return jnp.sin(x)
    def false_fun(x):
      return jnp.cos(x)
    def f(which, x):
      return olympus.lax.cond(which, true_fun, false_fun, x)
    hlo = module_to_string(olympus.jit(f).lower(True, 1.).compiler_ir())
    self.assertRegex(hlo, r'loc\(".*cond/branch_0_fun/cos"')
    self.assertRegex(hlo, r'loc\(".*cond/branch_1_fun/sin"')

  def test_argmax(self):
    def f(x):
      return jnp.argmax(x)
    hlo = module_to_string(olympus.jit(f).lower(jnp.arange(8.0)).compiler_ir())
    self.assertNotRegex(hlo, r'<.* at 0x[0-9a-fA-F]+>')

  @unittest.skip('b/352539562')
  def test_source_file_prefix_removal(self):

    def make_hlo():
      return module_to_string(
          olympus.jit(jnp.sin).lower(jnp.arange(8.0)).compiler_ir()
      )

    # Sanity check
    self.assertRegex(make_hlo(), r"[/\\]+tests[/\\]+metadata_test.py")

    with olympus_config.hlo_source_file_canonicalization_regex(r".*[\\/]+tests[/\\]+"):
      hlo = make_hlo()
      self.assertIn("metadata_test.py", hlo)
      self.assertNotRegex(hlo, r"tests[/\\]+")
      self.assertNotRegex(hlo, r"[/\\]+metadata_test.py")

    with olympus_config.hlo_source_file_canonicalization_regex("no_match_xxx"):
      hlo = make_hlo()
      self.assertRegex(hlo, r"[/\\]+tests[/\\]+metadata_test.py")

    with olympus_config.hlo_source_file_canonicalization_regex(".*"):
      hlo = make_hlo()
      self.assertNotIn("test.py", hlo)

    with olympus_config.hlo_source_file_canonicalization_regex("test"):
      hlo = make_hlo()
      self.assertRegex(hlo, r"[/\\]+s[/\\]+metadata_.py")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.OlympusTestLoader())
