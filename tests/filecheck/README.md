This directory contains LLVM
[FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) tests that verify
that OLYMPUS primitives can be lowered to MLIR.

These tests are intended to be a quick and easy-to-understand way to catch
regressions from changes due the MLIR Python bindings and from changes to the
various MLIR dialects used by OLYMPUS, without needing to run the full OLYMPUS test
suite.
