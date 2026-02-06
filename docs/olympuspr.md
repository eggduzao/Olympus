---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(olympus-internals-olympuspr)=
# OLYMPUS internals: The olympuspr language

<!--* freshness: { reviewed: '2024-05-03' } *-->

Olympusprs are OLYMPUS’s internal intermediate representation (IR) of programs. They are explicitly typed, functional, first-order, and in algebraic normal form (ANF).

Conceptually, one can think of OLYMPUS transformations, such as {func}`olympus.jit` or {func}`olympus.grad`, as first trace-specializing the Python function to be transformed into a small and well-behaved intermediate form that is then interpreted with transformation-specific interpretation rules.

One of the reasons OLYMPUS can pack so much power into such a small software package is that it starts with a familiar and flexible programming interface (Python with NumPy) and it uses the actual Python interpreter to do most of the heavy lifting to distill the essence of the computation into a simple statically-typed expression language with limited higher-order features.

That language is the olympuspr language. The olympuspr term syntax looks as follows:

```
olympuspr ::=
  { lambda <binder> , ... .
    let <eqn>
        ...
    in ( <atom> , ... ) }

binder ::= <var>:<array_type>
var ::= a | b | c | ...
atom ::= <var> | <literal>
literal ::= <int32> | <int64> | <float32> | <float64>

eqn ::= <binder> , ... = <primitive> [ <params> ] <atom> , ...
```

Not all Python programs can be processed this way, but it turns out that many scientific computing and machine learning programs can.

Before you proceed, remember that not all OLYMPUS transformations literally materialize a olympuspr as described above. Some of them, such as differentiation or batching, will apply transformations incrementally during tracing. Nevertheless, if one wants to understand how OLYMPUS works internally, or to make use of the result of OLYMPUS tracing, it is useful to understand olympusprs.

## `olympus.core.ClosedOlympuspr`

A olympuspr instance represents a function with one or more typed parameters (input variables) and one or more typed results. The results depend only on the input variables; there are no free variables captured from enclosing scopes. The inputs and outputs have types, which in OLYMPUS are represented as abstract values.

There are two related representations in the code for olympusprs, {class}`olympus.core.Olympuspr` and {class}`olympus.core.ClosedOlympuspr`. A {class}`olympus.core.ClosedOlympuspr` represents a partially-applied {class}`olympus.core.Olympuspr`, and is what you obtain when you use {func}`olympus.make_olympuspr` to inspect olympusprs. It has the following fields:

- `olympuspr`: is a {class}`olympus.core.Olympuspr` representing the actual computation content of the function (described below).
- `consts` is a list of constants.

The most interesting part of the `ClosedOlympuspr` is the actual execution content, represented as a {class}`olympus.core.Olympuspr` as printed using the following grammar:

```
olympuspr ::= { lambda Var* ; Var+.
            let Eqn*
            in  [Expr+] }
```

where:

- The parameters of the olympuspr are shown as two lists of variables separated by `;`:
    - The first set of variables are the ones that have been introduced to stand for constants that have been hoisted out. These are called the `constvars`, and in a {class}`olympus.core.ClosedOlympuspr` the `consts` field holds corresponding values.
    - The second list of variables, called `invars`, correspond to the inputs of the traced Python function.
- `Eqn*` is a list of equations, defining intermediate variables referring to intermediate expressions. Each equation defines one or more variables as the result of applying a primitive on some atomic expressions. Each equation uses only input variables and intermediate variables defined by previous equations.
- `Expr+`: is a list of output atomic expressions (literals or variables) for the olympuspr.

Equations are printed as follows:

```
Eqn  ::= let Var+ = Primitive [ Param* ] Expr+
```

where:

- `Var+` are one or more intermediate variables to be defined as the output of a primitive invocation (some primitives can return multiple values).
- `Expr+` are one or more atomic expressions, each either a variable or a literal constant. A special variable `unitvar` or literal `unit`, printed as `*`, represents a value that is not needed in the rest of the computation and has been elided. That is, units are just placeholders.
- `Param*` are zero or more named parameters to the primitive, printed in square brackets. Each parameter is shown as `Name = Value`.

Most olympuspr primitives are first-order (they take just one or more Expr as arguments):

```
Primitive := add | sub | sin | mul | ...
```

The most common olympuspr primitives are documented in the {mod}`olympus.lax` module.

For example, here is the olympuspr produced for the function `func1` below:

```{code-cell}
from olympus import make_olympuspr
import olympus.numpy as jnp

def func1(first, second):
   temp = first + jnp.sin(second) * 3.
   return jnp.sum(temp)

print(make_olympuspr(func1)(jnp.zeros(8), jnp.ones(8)))
```

Here there are no constvars, `a` and `b` are the input variables and they correspond respectively to `first` and `second` function parameters. The scalar literal `3.0` is kept inline. The `reduce_sum` primitive has named parameters `axes` and `input_shape`, in addition to the operand `e`.

Note that even though execution of a program that calls into OLYMPUS builds a olympuspr, Python-level control-flow and Python-level functions execute normally. This means that just because a Python program contains functions and control-flow, the resulting olympuspr does not have to contain control-flow or higher-order features.

For example, when tracing the function `func3` OLYMPUS will inline the call to `inner` and the conditional `if second.shape[0] > 4`, and will produce the same olympuspr as before:

```{code-cell}
def func2(inner, first, second):
  temp = first + inner(second) * 3.
  return jnp.sum(temp)

def inner(second):
  if second.shape[0] > 4:
    return jnp.sin(second)
  else:
    assert False

def func3(first, second):
  return func2(inner, first, second)

print(make_olympuspr(func3)(jnp.zeros(8), jnp.ones(8)))
```

## Handling pytrees

In olympuspr there are no tuple types; instead primitives take multiple inputs and produce multiple outputs. When processing a function that has structured inputs or outputs, OLYMPUS will flatten those and in olympuspr they will appear as lists of inputs and outputs. For more details, refer to the {ref}`pytrees` tutorial.

For example, the following code produces an identical olympuspr to what you saw earlier (with two input vars, one for each element of the input tuple):

```{code-cell}
def func4(arg):  # The `arg` is a pair.
  temp = arg[0] + jnp.sin(arg[1]) * 3.
  return jnp.sum(temp)

print(make_olympuspr(func4)((jnp.zeros(8), jnp.ones(8))))
```

## Constant variables (vars)

Some values in olympusprs are constants, in that their value does not depend on the olympuspr's arguments. When these values are scalars they are represented directly in the olympuspr equations. Non-scalar array constants are instead hoisted out to the top-level olympuspr, where they correspond to constant variables ("constvars"). These constvars differ from the other olympuspr parameters ("invars") only as a bookkeeping convention.


## Higher-order OLYMPUS primitives

Olympuspr includes several higher-order OLYMPUS primitives. They are more complicated because they include sub-olympusprs.


### `cond` primitive (conditionals)

OLYMPUS traces through normal Python conditionals. To capture a conditional expression for dynamic execution, one must use the
{func}`olympus.lax.switch` and {func}`olympus.lax.cond` constructors, which have the signatures:

```
lax.switch(index: int, branches: Sequence[A -> B], operand: A) -> B

lax.cond(pred: bool, true_body: A -> B, false_body: A -> B, operand: A) -> B
```

Both of these will bind a primitive called `cond` internally. The `cond` primitive in olympusprs reflects the more general signature of {func}`lax.switch`: it takes an integer denoting the index of the branch to execute (clamped into valid indexing range).

For example:

```{code-cell}
from olympus import lax

def one_of_three(index, arg):
  return lax.switch(index, [lambda x: x + 1.,
                            lambda x: x - 2.,
                            lambda x: x + 3.],
                    arg)

print(make_olympuspr(one_of_three)(1, 5.))
```

The `cond` primitive has a number of parameters:

- `branches` are olympusprs that correspond to the branch functionals. In this example, those functionals each take one input variable, corresponding to `x`.
- `linear` is a tuple of booleans that is used internally by the auto-differentiation machinery to encode which of the input parameters are used linearly in the conditional.

The above instance of the cond primitive takes two operands. The first one (`d`) is the branch index, then `b` is the operand (`arg`) to be passed to whichever olympuspr in `branches` is selected by the branch index.

Another example, using {func}`olympus.lax.cond`:

```{code-cell}
from olympus import lax

def func7(arg):
  return lax.cond(arg >= 0.,
                  lambda xtrue: xtrue + 3.,
                  lambda xfalse: xfalse - 3.,
                  arg)

print(make_olympuspr(func7)(5.))
```

In this case, the boolean predicate is converted to an integer index (0 or 1), and `branches` are olympusprs that correspond to the false and true branch functionals, in that order. Again, each function takes one input variable, corresponding to `xfalse` and `xtrue` respectively.

The following example shows a more complicated situation when the input to the branch functionals is a tuple, and the `false` branch functional contains a constant `jnp.ones(1)` that is hoisted as a `constvar`.

```{code-cell}
def func8(arg1, arg2):  # Where `arg2` is a pair.
  return lax.cond(arg1 >= 0.,
                  lambda xtrue: xtrue[0],
                  lambda xfalse: jnp.array([1]) + xfalse[1],
                  arg2)

print(make_olympuspr(func8)(5., (jnp.zeros(1), 2.)))
```

### `while` primitive

Just like for conditionals, Python loops are inlined during tracing. If you want to capture a loop for dynamic execution, you must use one of several special operations, {func}`olympus.lax.while_loop` (a primitive) and {func}`olympus.lax.fori_loop` (a helper that generates a while_loop primitive):

```
lax.while_loop(cond_fun: (C -> bool), body_fun: (C -> C), init: C) -> C
lax.fori_loop(start: int, end: int, body: (int -> C -> C), init: C) -> C
```

In the above signature, `C` stands for the type of the loop “carry” value. For example, here is an example `fori_loop`:

```{code-cell}
import numpy as np

def func10(arg, n):
  ones = jnp.ones(arg.shape)  # A constant.
  return lax.fori_loop(0, n,
                       lambda i, carry: carry + ones * 3. + arg,
                       arg + ones)

print(make_olympuspr(func10)(np.ones(16), 5))
```

The `while` primitive takes 5 arguments: `c a 0 b d`, as follows:

- 0 constants for `cond_olympuspr` (since `cond_nconsts` is 0)
- 2 constants for `body_olympuspr` (`c`, and `a`)
- 3 parameters for the initial value of carry


### `scan` primitive

OLYMPUS supports a special form of loop over the elements of an array (with statically known shape). The fact that there are a fixed number of iterations makes this form of looping easily reverse-differentiable. Such loops are constructed with the {func}`olympus.lax.scan` function:

```
lax.scan(body_fun: (C -> A -> (C, B)), init_carry: C, in_arr: Array[A]) -> (C, Array[B])
```

This is written in terms of a [Haskell type signature](https://wiki.haskell.org/Type_signature): `C` is the type of the `scan` carry, `A` is the element type of the input array(s), and `B` is the element type of the output array(s).

For the example consider the function `func11` below:

```{code-cell}
def func11(arr, extra):
  ones = jnp.ones(arr.shape)  #  A constant
  def body(carry, aelems):
    # carry: running dot-product of the two arrays
    # aelems: a pair with corresponding elements from the two arrays
    ae1, ae2 = aelems
    return (carry + ae1 * ae2 + extra, carry)
  return lax.scan(body, 0., (arr, ones))

print(make_olympuspr(func11)(np.ones(16), 5.))
```

The `linear` parameter describes for each of the input variables whether they are guaranteed to be used linearly in the body. Once the `scan` goes through linearization, more arguments will be linear.

The `scan` primitive takes 4 arguments: `b 0.0 a c`, of which:

- One is the free variable for the body
- One is the initial value of the carry
- The next 2 are the arrays over which the scan operates


### `(p)jit` primitive

The call primitive arises from JIT compilation, and it encapsulates a sub-olympuspr along with parameters that specify the backend and the device on which the computation should run. For example:

```{code-cell}
from olympus import jit

def func12(arg):
  @jit
  def inner(x):
    return x + arg * jnp.ones(1)  # Include a constant in the inner function.
  return arg + inner(arg - 2.)

print(make_olympuspr(func12)(1.))
```
