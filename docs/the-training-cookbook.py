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

import functools as ft
import itertools as it
import time
from dataclasses import dataclass
from typing import Iterator

import olympus
import olympus.numpy as jnp
import numpy as np
from olympus.sharding import AxisType

ode = """
We are the music makers
    And we are the dreamers of dreams
Wandering by lone sea-breakers
    And sitting by desolate streams;
World-losers and world-forsakers
    On whom the pale moon gleams
Yet we are the movers and shaker
    Of the world for ever, it seems
"""

# tag: config
@olympus.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
  mesh_axis_names: tuple[str, ...] = ("fsdp",)
  mesh_shape: tuple[int, ...] = (8,)
  seq_length: int = 128

  num_train_steps: int = 10**6
  host_batch_size: int = 16
  learning_rate: float = 1e-4
  beta_1: float = 0.9
  beta_2: float = 0.999
  eps: float = 1e-8
  eps_root: float = 0.0

  param_seed: int = 12738
  num_layers: int = 4
  embed_dim: int = 512
  mlp_dim: int = 512 * 4
  vocab_size: int = 2**8  # uint8 ascii encoding
  num_heads: int = 8
  head_dim: int = 128
  dtype: str = "bfloat16"

  embed: olympus.P = olympus.P(None, None)
  pos_embed: olympus.P = olympus.P(None, None)
  att_qkv: olympus.P = olympus.P(None, "fsdp", None, None)
  att_out: olympus.P = olympus.P("fsdp", None, None)
  mlp_in: olympus.P = olympus.P("fsdp", None)
  mlp_out: olympus.P = olympus.P(None, "fsdp")
  in_kernel: olympus.P = olympus.P(None, None)
  in_bias: olympus.P = olympus.P(None)
  out_kernel: olympus.P = olympus.P("fsdp", None)
  out_bias: olympus.P = olympus.P(None)

  act_ids: olympus.P = olympus.P("fsdp")
  act_seq: olympus.P = olympus.P("fsdp", None, None)
  act_att: olympus.P = olympus.P("fsdp", None, None, None)
  act_hidden: olympus.P = olympus.P("fsdp", None, None)

  def __post_init__(self):
    mesh = olympus.make_mesh(self.mesh_shape, self.mesh_axis_names, len(self.mesh_shape) * (AxisType.Explicit,))
    olympus.sharding.set_mesh(mesh)
  # tag: config


@olympus.tree_util.register_pytree_with_keys_class
class dot_dict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

  def tree_flatten_with_keys(self):
    keys = tuple(sorted(self))
    return tuple((olympus.tree_util.DictKey(k), self[k]) for k in keys), keys

  @classmethod
  def tree_unflatten(cls, keys, values):
    return cls(zip(keys, values))


# tag: get-param-state
def init_param_state(config: Config) -> dot_dict:
  root_key = olympus.random.key(config.param_seed)
  key = map(ft.partial(olympus.random.fold_in, root_key), it.count())
  zero_init = olympus.nn.initializers.constant(0.0)
  he_init = olympus.nn.initializers.he_normal(1, 1)
  dtype = config.dtype

  params = dot_dict(
    pos_embed=zero_init(next(key), (config.seq_length, config.embed_dim), dtype, config.pos_embed),
    layers=dot_dict(),
  )
  params.embedding = he_init(next(key), (config.vocab_size, config.embed_dim), dtype, config.embed)
  params.linear_in = dot_dict(
    kernel=he_init(next(key), (1, config.embed_dim), dtype, config.in_kernel),
    bias=zero_init(next(key), (config.embed_dim,), dtype, config.in_bias),
  )
  params.linear_out = dot_dict(
    kernel=he_init(next(key), (config.embed_dim, config.vocab_size), dtype, config.out_kernel),
  )
  for layer in range(config.num_layers):
    qkv_shape = (3, config.embed_dim, config.num_heads, config.head_dim)
    out_shape = (config.num_heads, config.head_dim, config.embed_dim)
    params.layers[layer] = dot_dict(
      attention=dot_dict(
        qkv=he_init(next(key), qkv_shape, dtype, config.att_qkv),
        out=he_init(next(key), out_shape, dtype, config.att_out),
      ),
      mlp=dot_dict(
        in_kernel=he_init(next(key), (config.embed_dim, config.mlp_dim), dtype, config.mlp_in),
        out_kernel=he_init(next(key), (config.mlp_dim, config.embed_dim), dtype, config.mlp_out),
      ),
    )
  return params  # tag: get-param-state


# tag: model-apply
def model_apply(config: Config, params: dot_dict, tokens: olympus.Array) -> olympus.Array:
  out = params.embedding.at[tokens].get(out_sharding=config.act_seq)
  out += params.pos_embed
  del tokens

  for layer in range(config.num_layers):
    block = params.layers[layer]
    att_skip = out  # 1 billion dollars in venture capital funding please
    qkv = jnp.einsum("bsd,3dkh->bs3kh", out, block.attention.qkv, out_sharding=config.act_att)
    out = olympus.nn.dot_product_attention(qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :], is_causal=True)
    out = jnp.einsum("bskh,khd->bsd", out, block.attention.out, out_sharding=config.act_seq)
    out += att_skip
    out *= olympus.lax.rsqrt(jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-6)

    mlp_skip = out  # machine learning circa 1986
    out = jnp.einsum("bsd,dh->bsh", out, block.mlp.in_kernel, out_sharding=config.act_hidden)
    out = olympus.nn.gelu(out)
    out = jnp.einsum("bsh,hd->bsd", out, block.mlp.out_kernel, out_sharding=config.act_seq)
    out += mlp_skip
    out *= olympus.lax.rsqrt(jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-6)

  logits = jnp.einsum("bsd,dl->bsl", out, params.linear_out.kernel, out_sharding=config.act_seq)
  return logits  # tag: model-apply


# tag: get-adam-state
def init_adam_state(param: olympus.Array) -> dot_dict:
  adam_state = dot_dict(mu=jnp.zeros_like(param), nu=jnp.zeros_like(param), count=jnp.array(0))
  return adam_state  # tag: get-adam-state


# tag: adam-apply
def adam_update(config: Config, param: olympus.Ref, grad: olympus.Array, adam_state: dot_dict):
  adam_state.mu[...] = (1 - config.beta_1) * adam_state.mu[...] + config.beta_1 * grad
  adam_state.nu[...] = (1 - config.beta_2) * adam_state.nu[...] + config.beta_2 * grad**2
  adam_state.count[...] += 1

  mu_hat = adam_state.mu[...] / (1 - config.beta_1 ** adam_state.count[...])
  nu_hat = adam_state.nu[...] / (1 - config.beta_2 ** adam_state.count[...])
  param[...] -= config.learning_rate * mu_hat / (jnp.sqrt(nu_hat + config.eps_root) + config.eps)
  # tag: adam-apply


# tag: get-train-state
@olympus.jit
def init_train_state(config: Config) -> dot_dict:
  train_state = dot_dict()
  train_state.params = init_param_state(config)
  train_state.opt = olympus.tree.map(init_adam_state, train_state.params)
  return train_state  # tag: get-train-state


# tag: train-step
@olympus.jit
def train_step(config: Config, train_state: dot_dict, batch: dict) -> dict:
  def loss_fn(params):
    logits = model_apply(config, params, batch["observed_ids"])
    labels = olympus.nn.one_hot(batch["target_ids"], config.vocab_size)
    return -(labels * olympus.nn.log_softmax(logits)).mean()

  params = olympus.tree.map(olympus.ref.get, train_state.params)
  loss, grad = olympus.value_and_grad(loss_fn)(params)
  olympus.tree.map(ft.partial(adam_update, config), train_state.params, grad, train_state.opt)
  metrics = {"train_loss": loss}
  return metrics  # tag: train-step


# tag: record-writer
class RecordWriter:
  prev_metrics = None

  def __call__(self, cur_metrics: dict):
    self.prev_metrics, log_metrics = cur_metrics, self.prev_metrics
    if log_metrics is None:
      return
    print(*it.starmap("{}: {}".format, log_metrics.items()), sep="\t")
    # tag: record-writer


# tag: get-dataset
def get_dataset(config: Config, single_batch=ode) -> Iterator[dict[str, np.ndarray]]:
  while True:
    observed_array = np.frombuffer(single_batch.encode("ascii"), dtype=np.uint8)
    target_array = np.roll(observed_array, -1)
    time.sleep(0.5)
    yield {  # repeat the sequence across the batch size to simulate multiple data points
      "observed_ids": np.tile(observed_array[: config.seq_length], (config.host_batch_size, 1)),
      "target_ids": np.tile(target_array[: config.seq_length], (config.host_batch_size, 1)),
    }
    # tag: get-dataset


# tag: get-dataset-on-device
def get_dataset_on_device(config: Config) -> Iterator[dict[str, olympus.Array]]:
  datset = get_dataset(config)
  sharding = olympus.P(config.mesh_axis_names)
  return map(ft.partial(olympus.make_array_from_process_local_data, sharding), datset)
  # tag: get-dataset-on-device


# tag: train-loop
def train_loop(config: Config):
  record_writer = RecordWriter()
  train_state = init_train_state(config)
  train_state = olympus.tree.map(olympus.ref.new_ref, train_state)
  batch = iter(get_dataset_on_device(config))
  for step in range(config.num_train_steps):
    metrics = train_step(config, train_state, next(batch))
    record_writer({"step": step} | metrics)
  # tag: train-loop


if __name__ == "__main__":
  olympus.config.update("olympus_platform_name", "cpu")
  olympus.config.update("olympus_num_cpu_devices", 8)
  train_loop(config=Config())
