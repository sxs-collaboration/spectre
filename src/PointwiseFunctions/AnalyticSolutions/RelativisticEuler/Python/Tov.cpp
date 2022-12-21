// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/Python/Tov.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/Tov.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

namespace py = pybind11;

namespace RelativisticEuler::Solutions::py_bindings {

void bind_tov(py::module& m) {
  py::enum_<TovCoordinates>(m, "TovCoordinates")
      .value("Schwarzschild", TovCoordinates::Schwarzschild)
      .value("Isotropic", TovCoordinates::Isotropic);
  py::class_<TovSolution>(m, "Tov")
      .def(py::init<const EquationsOfState::EquationOfState<true, 1>&, double,
                    TovCoordinates, double, double, double>(),
           py::arg("equation_of_state"), py::arg("central_mass_density"),
           py::arg("coordinate_system") = TovCoordinates::Schwarzschild,
           py::arg("log_enthalpy_at_outer_radius") = 0.,
           py::arg("absolute_tolerance") = 1.e-14,
           py::arg("relative_tolerance") = 1.e-14)
      .def("outer_radius", &TovSolution::outer_radius)
      .def("total_mass", &TovSolution::total_mass)
      .def("injection_energy", &TovSolution::injection_energy)
      .def("mass_over_radius",
           py::vectorize(&TovSolution::mass_over_radius<double>))
      .def("log_specific_enthalpy",
           py::vectorize(&TovSolution::log_specific_enthalpy<double>))
      .def("conformal_factor",
           py::vectorize(&TovSolution::conformal_factor<double>))
      .def_property_readonly("mass_over_radius_interpolant",
                             &TovSolution::mass_over_radius_interpolant)
      .def_property_readonly("log_specific_enthalpy_interpolant",
                             &TovSolution::log_specific_enthalpy_interpolant)
      .def_property_readonly("conformal_factor_interpolant",
                             &TovSolution::conformal_factor_interpolant);
}

}  // namespace RelativisticEuler::Solutions::py_bindings
