// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "Domain/Python/ElementId.hpp"
#include "Domain/Python/SegmentId.hpp"

namespace domain {

PYBIND11_MODULE(_PyDomain, m) {  // NOLINT
  py_bindings::bind_segment_id(m);
  py_bindings::bind_element_id(m);
}

}  // namespace domain
