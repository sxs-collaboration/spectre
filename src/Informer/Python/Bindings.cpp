// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "Informer/Python/InfoAtCompile.hpp"

PYBIND11_MODULE(_PyInformer, m) {  // NOLINT
  py_bindings::bind_info_at_compile(m);
}
