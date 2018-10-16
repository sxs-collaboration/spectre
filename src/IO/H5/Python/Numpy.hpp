// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

// These macros are required so that the NumPy API will work when used in
// multiple cpp files.  See
// https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL SPECTRE_IO_H5_PYTHON_BINDINGS
// Code is clean against Numpy 1.7.  See
// https://docs.scipy.org/doc/numpy-1.15.1/reference/c-api.deprecations.html
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
