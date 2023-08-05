// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/Python/LowerSpatialFourVelocity.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "PointwiseFunctions/Hydro/LowerSpatialFourVelocity.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_lowerVel(py::module& m) {
  m.def("LowerSpatialFourVelocityCompute",
        &hydro::Tags::LowerSpatialFourVelocityCompute::function,
        py::arg("result"), py::arg("spatial_velocity"),
        py::arg("spatial_metric"), py::arg("lorentz_factor"));
}
}  // namespace py_bindings
