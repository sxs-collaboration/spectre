// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Python/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Python/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Python/Spectral.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  Spectral::py_bindings::bind_logical_coordinates(m);
  Spectral::py_bindings::bind_spectral(m);
  py_bindings::bind_mesh(m);
  // Filtering
  m.def("exponential_filter", &Spectral::filtering::exponential_filter,
        py::arg("mesh"), py::arg("alpha"), py::arg("half_power"));
  m.def("zero_lowest_modes", &Spectral::filtering::zero_lowest_modes,
        py::arg("mesh"), py::arg("number_of_modes_to_zero"),
        py::return_value_policy::reference);
}
