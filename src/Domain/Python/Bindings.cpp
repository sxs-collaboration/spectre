// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "Domain/Python/Block.hpp"
#include "Domain/Python/BlockLogicalCoordinates.hpp"
#include "Domain/Python/Domain.hpp"
#include "Domain/Python/ElementId.hpp"
#include "Domain/Python/ElementLogicalCoordinates.hpp"
#include "Domain/Python/ElementMap.hpp"
#include "Domain/Python/FunctionsOfTime.hpp"
#include "Domain/Python/JacobianDiagnostic.hpp"
#include "Domain/Python/RadiallyCompressedCoordinates.hpp"
#include "Domain/Python/SegmentId.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

namespace domain {

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.Domain.CoordinateMaps");
  py_bindings::bind_block(m);
  py_bindings::bind_block_logical_coordinates(m);
  py_bindings::bind_domain(m);
  py_bindings::bind_element_id(m);
  py_bindings::bind_element_logical_coordinates(m);
  py_bindings::bind_element_map(m);
  py_bindings::bind_functions_of_time(m);
  py_bindings::bind_jacobian_diagnostic(m);
  py_bindings::bind_radially_compressed_coordinates(m);
  py_bindings::bind_segment_id(m);
}

}  // namespace domain
