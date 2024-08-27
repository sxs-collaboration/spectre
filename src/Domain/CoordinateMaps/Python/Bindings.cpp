// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Python/CoordinateMap.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

namespace domain {

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::enum_<CoordinateMaps::Distribution>(m, "Distribution")
      .value("Linear", CoordinateMaps::Distribution::Linear)
      .value("Equiangular", CoordinateMaps::Distribution::Equiangular)
      .value("Logarithmic", CoordinateMaps::Distribution::Logarithmic)
      .value("Inverse", CoordinateMaps::Distribution::Inverse)
      .value("Projective", CoordinateMaps::Distribution::Projective);
  py::class_<CoordinateMaps::DistributionAndSingularityPosition>(
      m, "DistributionAndSingularityPosition")
      .def(py::init<CoordinateMaps::Distribution, std::optional<double>>(),
           py::arg("distribution"),
           py::arg("singularity_position") = std::nullopt);
  // Order is important: The base class `CoordinateMap` needs to have its
  // bindings set up before the derived classes
  py_bindings::bind_coordinate_map(m);
}

}  // namespace domain
