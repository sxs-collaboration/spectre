// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Python/EquationOfState.hpp"

#include <pybind11/pybind11.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

namespace py = pybind11;

namespace EquationsOfState::py_bindings {

void bind_equation_of_state(py::module& m) {
  using EosType = EquationOfState<true, 1>;
  py::class_<EosType>(m, "RelativisticEquationOfState1D")
      .def(
          "pressure_from_density",
          [](const EosType& eos, const Scalar<DataVector>& rest_mass_density) {
            return eos.pressure_from_density(rest_mass_density);
          },
          py::arg("rest_mass_density"))
      .def(
          "rest_mass_density_from_enthalpy",
          [](const EosType& eos, const Scalar<DataVector>& specific_enthalpy) {
            return eos.rest_mass_density_from_enthalpy(specific_enthalpy);
          },
          py::arg("specific_enthalpy"))
      .def(
          "specific_internal_energy_from_density",
          [](const EosType& eos, const Scalar<DataVector>& rest_mass_density) {
            return eos.specific_internal_energy_from_density(rest_mass_density);
          },
          py::arg("rest_mass_density"))
      .def(
          "temperature_from_density",
          [](const EosType& eos, const Scalar<DataVector>& rest_mass_density) {
            return eos.temperature_from_density(rest_mass_density);
          },
          py::arg("rest_mass_density"))
      .def(
          "temperature_from_specific_internal_energy",
          [](const EosType& eos,
             const Scalar<DataVector>& specific_internal_energy) {
            return eos.temperature_from_specific_internal_energy(
                specific_internal_energy);
          },
          py::arg("specific_internal_energy"))
      .def(
          "chi_from_density",
          [](const EosType& eos, const Scalar<DataVector>& rest_mass_density) {
            return eos.chi_from_density(rest_mass_density);
          },
          py::arg("rest_mass_density"))
      .def(
          "kappa_times_p_over_rho_squared_from_density",
          [](const EosType& eos, const Scalar<DataVector>& rest_mass_density) {
            return eos.kappa_times_p_over_rho_squared_from_density(
                rest_mass_density);
          },
          py::arg("rest_mass_density"));
}

}  // namespace EquationsOfState::py_bindings
