// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

namespace py = pybind11;

namespace gr::Solutions::py_bindings {

void bind_tov(py::module& m) {  // NOLINT
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
           py::vectorize(&TovSolution::conformal_factor<double>));
}

}  // namespace gr::Solutions::py_bindings
