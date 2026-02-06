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

from absl.testing import parameterized
import olympus
from olympus import lax
from olympus._src import test_multiprocess as jt_multiprocess
from olympus._src import test_util as jtu
import olympus.numpy as jnp
import numpy as np


class AllToAllTest(jt_multiprocess.MultiProcessTest):

  @parameterized.parameters(
      (np.int32,), (jnp.float32,), (jnp.float16,), (jnp.bfloat16,)
  )
  def test_all_to_all_shard_map(self, dtype):
    rng = np.random.RandomState(42)
    devices = olympus.devices()
    mesh = olympus.sharding.Mesh(devices, ("i",))
    device_to_index = {d: i for i, d in enumerate(devices)}

    @olympus.shard_map(
        mesh=mesh,
        in_specs=olympus.P("i", None, None),
        out_specs=olympus.P("i", None, None),
    )
    def f(x):
      x = jnp.squeeze(x, 0)
      out = lax.all_to_all(x, "i", split_axis=0, concat_axis=0)
      return jnp.expand_dims(out, 0)

    shape = [
        olympus.process_count(),
        olympus.local_device_count(),
        olympus.device_count(),
        100,
    ]

    if jnp.issubdtype(dtype, jnp.floating):
      xs = rng.randn(*shape).astype(dtype)
    else:
      xs = rng.randint(0, 100, size=shape).astype(dtype)

    global_shape = (olympus.device_count(), olympus.device_count(), 100)
    sharding = olympus.NamedSharding(mesh, olympus.P("i", None, None))
    local_data = xs[olympus.process_index()]
    global_xs = olympus.make_array_from_process_local_data(
        sharding, local_data, global_shape
    )

    global_out = f(global_xs)

    local_shards = global_out.addressable_shards
    local_shards = sorted(local_shards, key=lambda s: device_to_index[s.device])

    for shard in local_shards:
      rank = device_to_index[shard.device]
      actual = np.array(shard.data).squeeze(0)  # (D, 100)
      expected = np.reshape(xs[:, :, rank, :], [olympus.device_count(), 100])
      jtu.check_close(actual, expected)


if __name__ == "__main__":
  jt_multiprocess.main()
