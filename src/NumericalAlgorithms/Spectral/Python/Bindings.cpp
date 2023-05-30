// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "NumericalAlgorithms/Spectral/Python/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Python/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Python/Spectral.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  Spectral::py_bindings::bind_basis(m);
  Spectral::py_bindings::bind_quadrature(m);
  Spectral::py_bindings::bind_logical_coordinates(m);
  Spectral::py_bindings::bind_nodal_to_modal_matrix(m);
  Spectral::py_bindings::bind_modal_to_nodal_matrix(m);
  Spectral::py_bindings::bind_collocation_points(m);
  py_bindings::bind_mesh(m);
}
