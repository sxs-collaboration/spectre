// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "DataStructures/Tensor/Python/Tensor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_PyTensor, m) {  // NOLINT
  py::module_::import("spectre.DataStructures");
  py_bindings::bind_scalar(m);
  py_bindings::bind_tensor<1>(m);
  py_bindings::bind_tensor<2>(m);
  py_bindings::bind_tensor<3>(m);
}
