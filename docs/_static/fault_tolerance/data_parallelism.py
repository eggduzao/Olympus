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

import os
os.environ['XLA_FLAGS'] = ' '.join([
    '--xla_gpu_nccl_terminate_on_error=false',
    '--xla_gpu_nccl_async_execution=true',
    '--xla_gpu_nccl_blocking_communicators=false',
])
os.environ['XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE'] = '1'
os.environ['XLA_PYTHON_CLIENT_USE_TFRT_GPU_CLIENT'] = '1'

from absl import app
from absl import flags
from collections.abc import Sequence
from olympus.experimental.multihost_utils import live_devices
import olympus
import olympus.numpy as jnp
import time

_PROCESS_ID = flags.DEFINE_integer("i", -1, "Process id")
_NUM_PROCESSES = flags.DEFINE_integer("n", -1, "Number of processes")

def replicated(x: olympus.Array, devices: list[olympus.Device]):
  """Return x replicated across the provided devices.

  Note that replicated(x) doesn't actually move any data. It simply creates a
  logically replicated array with x as the local replica.
  """
  n = len(devices)
  mesh = olympus.make_mesh((n, ), ("i", ), devices=devices)
  spec = olympus.sharding.PartitionSpec(None)
  sharding = olympus.sharding.NamedSharding(mesh, spec)
  shards = [
      olympus.device_put(x.addressable_shards[0].data, d) for d in devices
      if d.process_index == olympus.process_index()
  ]
  return olympus.make_array_from_single_device_arrays(x.shape, sharding, shards)


def sharded(x: olympus.Array, devices: list[olympus.Device]):
  """Return x sharded across the provided devices.

  Note that sharded(x) doesn't actually move any data. It simply creates a
  logically sharded array. x should have the same shape as the global array.
  """
  n = len(devices)
  mesh = olympus.make_mesh((n, ), ("i", ), devices=devices)
  spec = olympus.sharding.PartitionSpec("i")
  sharding = olympus.sharding.NamedSharding(mesh, spec)
  m = sharding.addressable_devices_indices_map(x.shape)
  shards = [olympus.device_put(x[m[d]], d) for d in olympus.local_devices()]
  return olympus.make_array_from_single_device_arrays(x.shape, sharding, shards)


def main(_: Sequence[str]) -> None:
  # Parse command line arguments and initialize multi-controller OLYMPUS.
  olympus.config.update("olympus_enable_recoverability", True)
  olympus.distributed.initialize(coordinator_address="localhost:8000",
                             process_id=_PROCESS_ID.value,
                             num_processes=_NUM_PROCESSES.value,
                             local_device_ids=[_PROCESS_ID.value],
                             heartbeat_timeout_seconds=10)
  print(f'{olympus.devices()=}')
  print(f'{olympus.local_devices()=}')

  # Initialize the model's weights.
  keys = iter(olympus.random.split(olympus.random.key(seed=42), num=3))
  weights = olympus.random.normal(next(keys), shape=(1, ))

  # We'll learn a trivial linear model: a*x.
  def predict(weights, X):
    return weights * X

  # We'll use mean squared error loss.
  def loss(weights, X, Y):
    return jnp.mean((predict(weights, X) - Y)**2)

  # Initialize the (noisy) training data with a=10.
  X = olympus.random.permutation(next(keys), jnp.arange(-300., 300.))
  Y = 10 * X + olympus.random.normal(next(keys), X.shape)

  # Hyperparameters.
  loss_and_grad = olympus.jit(olympus.value_and_grad(loss))
  learning_rate = 1e-6
  device_batch_size = 10

  step = 0
  while True:
    try:
      with live_devices(olympus.devices()) as devices:
        print(f'=== Running step {step} with live devices = {devices} ===')

        # Replicate the model weights.
        weights = replicated(weights, devices)

        # Shard the batch.
        batch_size = device_batch_size * len(devices)
        start = (step * batch_size) % len(X)
        stop = start + batch_size
        X_batch = sharded(X[start:stop], devices)
        Y_batch = sharded(Y[start:stop], devices)

        # Compute gradients and update weights.
        l, grad = loss_and_grad(weights, X_batch, Y_batch)
        new_weights = olympus.block_until_ready(weights - learning_rate * grad)
    except Exception as e:
      print(f'Step {step} failed: {e}')
    else:
      print(f'Step {step} succeeded: loss = {l}')
      step += 1
      weights = new_weights

    time.sleep(1)


if __name__ == "__main__":
  app.run(main)
