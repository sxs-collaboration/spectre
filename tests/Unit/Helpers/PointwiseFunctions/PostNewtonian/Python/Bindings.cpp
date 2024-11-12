// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/PointwiseFunctions/PostNewtonian/BinaryTrajectories.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module::import("spectre.DataStructures");
  py::module::import("spectre.DataStructures.Tensor");
  py::class_<BinaryTrajectories>(m, "BinaryTrajectories")
      .def(py::init<double, const std::array<double, 3>&, bool>(),
           py::arg("initial_separation"),
           py::arg("center_of_mass_velocity") =
               std::array<double, 3>{{0.0, 0.0, 0.0}},
           py::arg("newtonian") = false)
      .def("separation", &BinaryTrajectories::separation<double>,
           py::arg("time"))
      .def("separation", &BinaryTrajectories::separation<DataVector>,
           py::arg("time"))
      .def("orbital_frequency", &BinaryTrajectories::orbital_frequency<double>,
           py::arg("time"))
      .def("orbital_frequency",
           &BinaryTrajectories::orbital_frequency<DataVector>, py::arg("time"))
      .def("angular_velocity", &BinaryTrajectories::angular_velocity<double>,
           py::arg("time"))
      .def("angular_velocity",
           &BinaryTrajectories::angular_velocity<DataVector>, py::arg("time"))
      .def("positions", &BinaryTrajectories::positions<double>, py::arg("time"))
      .def("positions", &BinaryTrajectories::positions<DataVector>,
           py::arg("time"))
      .def("positions_no_expansion",
           &BinaryTrajectories::positions_no_expansion<double>, py::arg("time"))
      .def("positions_no_expansion",
           &BinaryTrajectories::positions_no_expansion<DataVector>,
           py::arg("time"));
}
