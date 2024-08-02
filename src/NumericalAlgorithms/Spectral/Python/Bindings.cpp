// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
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
  // Projection
  py::enum_<Spectral::ChildSize>(m, "ChildSize")
      .value("Uninitialized", Spectral::ChildSize::Uninitialized)
      .value("Full", Spectral::ChildSize::Full)
      .value("UpperHalf", Spectral::ChildSize::UpperHalf)
      .value("LowerHalf", Spectral::ChildSize::LowerHalf);
  m.def("projection_matrix_parent_to_child",
        static_cast<const Matrix& (*)(const Mesh<1>&, const Mesh<1>&,
                                      Spectral::ChildSize)>(
            &Spectral::projection_matrix_parent_to_child),
        py::arg("parent_mesh"), py::arg("child_mesh"), py::arg("size"));
  m.def("projection_matrix_child_to_parent",
        static_cast<const Matrix& (*)(const Mesh<1>&, const Mesh<1>&,
                                      Spectral::ChildSize, bool)>(
            &Spectral::projection_matrix_child_to_parent),
        py::arg("child_mesh"), py::arg("parent_mesh"), py::arg("size"),
        py::arg("operand_is_massive"));
}
