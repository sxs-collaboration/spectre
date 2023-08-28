// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Python/Enthalpy.hpp"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "PointwiseFunctions/Hydro/EquationsOfState/Enthalpy.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"
// low_density_eos uses Spectral and Polytropic Fluid type EoS.

namespace py = pybind11;

namespace EquationsOfState::py_bindings {

template <typename LowDensityEoS>
// NOLINTNEXTLINE(google-runtime-references)
void bind_enthalpy_impl(py::module& m, const std::string& name) {
  py::class_<Enthalpy<LowDensityEoS>, EquationOfState<true, 1>>(
    m, name.c_str()).def(
      py::init<double, double, double, double, std::vector<double>,
               std::vector<double>, std::vector<double>, LowDensityEoS,
               double>(),
      py::arg("reference_density"), py::arg("max_density"),
      py::arg("min_density"), py::arg("trig_scale"),
      py::arg("polynomial_coefficients"), py::arg("sin_coefficients"),
      py::arg("cos_coefficients"), py::arg("low_density_eos"),
      py::arg("transition_delta_epsilon"));
}

void bind_enthalpy(py::module& m) {
  bind_enthalpy_impl<PolytropicFluid<true>>(m, "EnthalpyPolytropicFluid");
  bind_enthalpy_impl<Spectral>(m, "EnthalpySpectral");
  bind_enthalpy_impl<Enthalpy<Spectral>>(m, "EnthalpyEnthalpySpectral");
  bind_enthalpy_impl<Enthalpy<Enthalpy<Spectral>>>(
      m, "EnthalpyEnthalpyEnthalpySpectral");
}

}  // namespace EquationsOfState::py_bindings
