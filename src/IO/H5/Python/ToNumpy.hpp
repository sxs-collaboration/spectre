// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <Python.h>
#include <array>
#include <cstdlib>

#include "DataStructures/Matrix.hpp"
// IWYU pragma: no_include <numpy/ndarrayobject.h>
// IWYU pragma: no_include <numpy/ndarraytypes.h>

namespace py_bindings {
/// Convert Matrix to a Numpy Array. Always creates a copy.
// We use `malloc` instead of `new` because we tell NumPy it needs to free the
// memory and NumPy uses `free`, not `delete`.
inline PyObject* to_numpy(const Matrix& matrix) {
  auto* c_style_data = static_cast<double*>(
      malloc(sizeof(double) * (matrix.rows()) * matrix.columns()));
  for (size_t i = 0; i < matrix.rows(); ++i) {
    for (size_t j = 0; j < matrix.columns(); ++j) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      c_style_data[j + i * matrix.columns()] = matrix(i, j);
    }
  }
  std::array<long, 2> dims{
      {static_cast<long>(matrix.rows()), static_cast<long>(matrix.columns())}};
  // clang-tidy: C-style cast done implictly with Python
  // NOLINTNEXTLINE
  PyObject* npy_array = PyArray_SimpleNewFromData(2, dims.data(), NPY_DOUBLE,
                                                  c_style_data);  // NOLINT

  // The `reinterpret_cast` is intentional because we know the pointer actually
  // points to an object of type `PyArrayObject` and we need to access it in
  // that manner.

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* npy_array_obj = reinterpret_cast<PyArrayObject*>(npy_array);
  PyArray_ENABLEFLAGS(npy_array_obj, NPY_ARRAY_OWNDATA);
  return npy_array;
}
}  // namespace py_bindings
