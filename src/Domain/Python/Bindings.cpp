// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "Domain/Python/Domain.hpp"
#include "Domain/Python/ElementId.hpp"
#include "Domain/Python/FunctionsOfTime.hpp"
#include "Domain/Python/SegmentId.hpp"

namespace py = pybind11;

namespace domain {

PYBIND11_MODULE(_PyDomain, m) {  // NOLINT
  py::module_::import("spectre.DataStructures");
  py_bindings::bind_domain(m);
  py_bindings::bind_element_id(m);
  py_bindings::bind_functions_of_time(m);
  py_bindings::bind_segment_id(m);
}

}  // namespace domain
