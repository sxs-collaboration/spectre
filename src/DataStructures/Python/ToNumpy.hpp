// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <Python.h>

/// \cond
class Matrix;
/// \endcond

namespace py_bindings {
/// Convert Matrix to a Numpy Array. Always creates a copy.
PyObject* to_numpy(const Matrix& matrix);
}  // namespace py_bindings
