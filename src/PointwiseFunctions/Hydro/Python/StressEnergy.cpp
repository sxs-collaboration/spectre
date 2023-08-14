// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/Python/StressEnergy.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_stress_energy(py::module& m) {
  m.def("energy_density", &hydro::energy_density<double>, py::arg("result"),
        py::arg("rest_mass_density"), py::arg("specific_enthalpy"),
        py::arg("pressure"), py::arg("lorentz_factor"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("comoving_magnetic_field_squared"));
  m.def("energy_density", &hydro::energy_density<DataVector>, py::arg("result"),
        py::arg("rest_mass_density"), py::arg("specific_enthalpy"),
        py::arg("pressure"), py::arg("lorentz_factor"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("comoving_magnetic_field_squared"));
  m.def("momentum_density", &hydro::momentum_density<double>, py::arg("result"),
        py::arg("rest_mass_density"), py::arg("specific_enthalpy"),
        py::arg("spatial_velocity"), py::arg("lorentz_factor"),
        py::arg("magnetic_field"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("comoving_magnetic_field_squared"));
  m.def("momentum_density", &hydro::momentum_density<DataVector>,
        py::arg("result"), py::arg("rest_mass_density"),
        py::arg("specific_enthalpy"), py::arg("spatial_velocity"),
        py::arg("lorentz_factor"), py::arg("magnetic_field"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("comoving_magnetic_field_squared"));
  m.def("stress_trace", &hydro::stress_trace<double>, py::arg("result"),
        py::arg("rest_mass_density"), py::arg("specific_enthalpy"),
        py::arg("pressure"), py::arg("spatial_velocity_squared"),
        py::arg("lorentz_factor"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("comoving_magnetic_field_squared"));
  m.def("stress_trace", &hydro::stress_trace<DataVector>, py::arg("result"),
        py::arg("rest_mass_density"), py::arg("specific_enthalpy"),
        py::arg("pressure"), py::arg("spatial_velocity_squared"),
        py::arg("lorentz_factor"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("comoving_magnetic_field_squared"));
}
}  // namespace py_bindings
