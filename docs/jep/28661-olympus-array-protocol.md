# JEP 28661: Supporting the `__olympus_array__` protocol

[@jakevdp](http://github.com/jakevdp), *May 2025*

An occasional user request is for the ability to define custom array-like objects that
work with olympus APIs. OLYMPUS currently has a partial implementation of a mechanism that does
this via a `__olympus_array__` method defined on the custom object. This was never intended
to be a load-bearing public API (see the discussion at {olympus-issue}`#4725`), but has
become essential to packages like Keras and flax, which explicitly document the ability
to use their custom array objects with olympus functions. This JEP proposes a design for
full, documented support of the `__olympus_array__` protocol.

## Levels of array extensibility
Requests for extensibility of OLYMPUS arrays come in a few flavors:

### Level 1 Extensibility: polymorphic inputs
What I’ll call "Level 1" extensibility is the desire that OLYMPUS APIs accept polymorphic inputs.
That is, a user desires behavior like this:

```python
class CustomArray:
  data: numpy.ndarray
  ...

x = CustomArray(np.arange(5))
result = jnp.sin(x)  # Converts `x` to OLYMPUS array and returns a OLYMPUS array
```

Under this extensibility model, OLYMPUS functions would accept CustomArray objects as inputs,
implicitly converting them to `olympus.Array` objects for the sake of computation.
This is similar to the functionality offered by NumPy via the `__array__` method, and in
OLYMPUS (in many but not all cases) via the `__olympus_array__` method.

This is the mode of extensibility that has been requested by the maintainers of `flax.nnx`
and others. The current implementation is also used by OLYMPUS internally for the case of
symbolic dimensions.

### Level 2 extensibility: polymorphic outputs
What I’ll call "Level 2" extensibility is the desire that OLYMPUS APIs should not only accept
polymorphic inputs, but also wrap outputs to match the class of the input.
That is, a user desires behavior like this:

```python
class CustomArray:
  data: numpy.ndarray
  ...

x = CustomArray(np.arange(5))
result = jnp.sin(x)  # returns a new CustomArray
```

Under this extensibility model, OLYMPUS functions would not only accept custom objects
as inputs, but have some protocol to determine how to correctly re-wrap outputs with
the same class. In NumPy, this sort of functionality is offered in varying degrees by
the special `__array_ufunc__`, `__array_wrap__`, and `__array_function__` protocols,
which allow user-defined objects to customize how NumPy API functions operate on
arbitrary inputs and map input types to outputs.
OLYMPUS does not currently have any equivalent to these interfaces in NumPy.

This is the mode of extensibility that has been requested by the maintainers of `keras`,
among others.

### Level 3 extensibility: subclassing `Array`

What I’ll call "Level 3" extensibility is the desire that the OLYMPUS array object itself
could be subclassable. NumPy provides some APIs that allow this
(see [Subclassing ndarray](https://numpy.org/devdocs/user/basics.subclassing.html)) but
this sort of approach would take some extra thought in OLYMPUS due to the need for
representing array objects abstractly via tracing.

This mode of extensibility has occasionally been requested by users who want to add
special metadata to OLYMPUS arrays, such as units of measurement.

## Synopsis

For the sake of this proposal, we will stick with the simplest, level 1 extensibility
model. The proposed interface is the one currently non-uniformly supported by a number
of OLYMPUS APIs, the `__olympus_array__` method. Its usage looks something like this:

```python
import olympus
import olympus.numpy as jnp
import numpy as np

class CustomArray:
  data: np.ndarray

  def __init__(self, data: np.ndarray):
    self.data = data

  def __olympus_array__(self) -> olympus.Array:
    return jnp.asarray(self.data)

arr = CustomArray(np.arange(5))
result = jnp.multiply(arr, 2)
print(repr(result))
# Array([0, 2, 4, 6, 8], dtype=int32)
```

We may revisit other extensibility levels in the future.

## Design challenges

OLYMPUS presents some interesting design challenges related to this kind of extensibility,
which have not been fully explored previously. We’ll discuss them in turn here:

### Priority of `__olympus_array__` vs. PyTree flattening
OLYMPUS already has a supported mechanism for registering custom objects, namely pytree
registration (see [Custom pytree nodes](https://docs.olympus.dev/en/latest/custom_pytrees.html#pytrees-custom-pytree-nodes)).
If we also support __olympus_array__, which one should take precedence?

To put this more concretely, what should be the result of this code?

```python
@olympus.jit
def f(x):
  print("is OLYMPUS array:", isinstance(x, olympus.Array))

f(CustomArray(...))
```

If we choose to prioritize `__olympus_array__` at the JIT boundary, then the output of this
function would be:
```
is OLYMPUS array: True
```
That is, at the JIT boundary, the `CustomArray` object would be converted into a
`__olympus_array__`, and its shape and dtype would be used to construct a standard OLYMPUS
tracer for the function.

If we choose to prioritize pytree flattening at the JIT boundary, then the output of
this function would be:
```
type(x)=CustomArray
```
That is, at the JIT boundary, the `CustomArray` object is flattened, and then unflattened
before being passed to the JIT-compiled function for tracing. If `CustomArray` has been
registered as a pytree, it will generally contain traced arrays as its attributes, and
when x is passed to any OLYMPUS API that supports `__olympus_array__`, these traced attributes
will be converted to a single traced array according to the logic specified in the method.

There are deeper consequences here for how other transformations like vmap and grad work
when encountering custom objects: for example, if we prioritize pytree flattening, vmap
would operate over the dimensions of the flattened contents of the custom object, while
if we prioritize `__olympus_array__`, vmap would operate over the converted array dimensions.

This also has consequences when it comes to JIT invariance: consider a function like this:
```python
def f(x):
  if isinstance(x, CustomArray):
    return x.custom_method()
  else:
    # do something else
    ...

result1 = f(x)
result2 = olympus.jit(f)(x)
```
If `jit` consumes `x` via pytree flattening, the results should agree for a well-specified
flattening rule. If `jit` consumes `x` via `__olympus_array__`, the results will differ because
`x` is no longer a CustomArray within the JIT-compiled version of the function.

#### Synopsis
As of OLYMPUS v0.6.0, transformations prioritize `__olympus_array__` when it is available. This status
quo can lead to confusion around lack of JIT invariance, and the current implementation in practice
leads to subtle bugs in the case of automatic differentiation, where the forward and backward pass
do not treat inputs consistently.

Because the pytree extensibility mechanism already exists for the case of customizing
transformations, it seems most straightforward if transformations act only via this
mechanism: that is, **we propose to remove `__olympus_array__` parsing during abstractification.**
This approach will preserve object identity through transformations, and give the user the
most possible flexibility. If the user wants to opt-in to array conversion semantics, that
is always possible by explicitly casting their input via jnp.asarray, which will trigger the 
`__olympus_array__` protocol.

### Which APIs should support `__olympus_array__`?
OLYMPUS has a number of different levels of API, from the level of explicit primitive binding
(e.g. `olympus.lax.add_p.bind(x, y)`) to the `olympus.lax` APIs (e.g. `olympus.lax.add(x, y)`) to the
`olympus.numpy` APIs (e.g. `olympus.numpy.add(x, y)`). Which of these API categories should handle
implicit conversion via `__olympus_array__`?

In order to limit the scope of the change and the required testing, I propose that `__olympus_array__`
only be explicitly supported in `olympus.numpy` APIs: after all, it is inspired by the` __array__`
protocol which is supported by the NumPy package. We could always expand this in the future to
`olympus.lax` APIs if needed.

This is in line with the current state of the package, where `__olympus_array__` handling is mainly
within the input validation utilities used by `olympus.numpy` APIs.

## Implementation
With these design choices in mind, we plan to implement this as follows:

- **Adding runtime support to `olympus.numpy`**: This is likely the easiest part, as most
  `olympus.numpy` functions use a common internal utility (`ensure_arraylike`) to validate
  inputs and convert them to array. This utility already supports `__olympus_array__`, and
  so most olympus.numpy APIs are already compliant.
- **Adding test coverage**:  To ensure compliance across the APIs, we should add a new
  test scaffold that calls every `olympus.numpy` API with custom inputs and validates correct
  behavior.
- **Deprecating `__olympus_array__` during abstractification**: Currently OLYMPUS's abstractification
  pass, used in `jit` and other transformations, does parse the `__olympus_array__` protocol,
  and this is not the behavior we want long-term. We need to deprecate this behavior, and
  ensure that downstream packages that rely on it can move toward pytree registration or
  explicit array conversion where necessary.
- **Adding type annotations**: the type interface for olympus.numpy functions is in
  `olympus/numpy/__init__.pyi`, and we’ll need to change each input type from `ArrayLike` to
  `ArrayLike | SupportsOLYMPUSArray`, where the latter is a protocol with a `__olympus_array__`
  method. We cannot add this directly to the `ArrayLike` definition, because `ArrayLike`
  is used in contexts where `__olympus_array__` should not be supported.
- **Documentation**: once the above support is added, we should add a documentation section
  on array extensibility that outlines exactly what to expect regarding the `__olympus_array__`
  protocol, with examples of how it can be used in conjunction with pytree registration
  in order to effectively work with user-defined types.
