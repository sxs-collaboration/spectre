// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "DataStructures/Python/DataVector.hpp"
#include "DataStructures/Python/Matrix.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py_bindings::bind_datavector(m);
  py_bindings::bind_matrix(m);
}
