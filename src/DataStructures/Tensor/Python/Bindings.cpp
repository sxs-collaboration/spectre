// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "DataStructures/Tensor/Python/Tensor.hpp"

PYBIND11_MODULE(_PyTensor, m) {  // NOLINT
  py_bindings::bind_tensor(m);
}
