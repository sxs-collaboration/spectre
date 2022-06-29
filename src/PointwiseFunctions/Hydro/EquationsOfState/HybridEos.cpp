// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"

#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace EquationsOfState {
template <typename ColdEquationOfState>
HybridEos<ColdEquationOfState>::HybridEos(
    ColdEquationOfState cold_eos, const double thermal_adiabatic_index)
    : cold_eos_(std::move(cold_eos)),
      thermal_adiabatic_index_(thermal_adiabatic_index) {}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEquationOfState>,
                                     HybridEos<ColdEquationOfState>, double, 2)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEquationOfState>,
                                     HybridEos<ColdEquationOfState>, DataVector,
                                     2)

template <typename ColdEquationOfState>
HybridEos<ColdEquationOfState>::HybridEos(CkMigrateMessage* msg)
    : EquationOfState<is_relativistic, 2>(msg) {}

template <typename ColdEquationOfState>
void HybridEos<ColdEquationOfState>::pup(PUP::er& p) {
  EquationOfState<is_relativistic, 2>::pup(p);
  p | cold_eos_;
  p | thermal_adiabatic_index_;
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType>
HybridEos<ColdEquationOfState>::pressure_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy) const {
  return Scalar<DataType>{
      get(cold_eos_.pressure_from_density(rest_mass_density)) +
      get(rest_mass_density) * (thermal_adiabatic_index_ - 1.0) *
          (get(specific_internal_energy) -
           get(cold_eos_.specific_internal_energy_from_density(
               rest_mass_density)))};
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType>
HybridEos<ColdEquationOfState>::pressure_from_density_and_enthalpy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy) const {
  if constexpr (ColdEquationOfState::is_relativistic) {
    return Scalar<DataType>{
        (get(cold_eos_.pressure_from_density(rest_mass_density)) +
         get(rest_mass_density) * (thermal_adiabatic_index_ - 1.0) *
             (get(specific_enthalpy) - 1.0 -
              get(cold_eos_.specific_internal_energy_from_density(
                  rest_mass_density)))) /
        thermal_adiabatic_index_};
  } else {
    return Scalar<DataType>{
        (get(cold_eos_.pressure_from_density(rest_mass_density)) +
         get(rest_mass_density) * (thermal_adiabatic_index_ - 1.0) *
             (get(specific_enthalpy) -
              get(cold_eos_.specific_internal_energy_from_density(
                  rest_mass_density)))) /
        thermal_adiabatic_index_};
  }
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType> HybridEos<ColdEquationOfState>::
    specific_internal_energy_from_density_and_pressure_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& pressure) const {
  return Scalar<DataType>{
      get(cold_eos_.specific_internal_energy_from_density(rest_mass_density)) +
      1.0 / (thermal_adiabatic_index_ - 1.0) *
          (get(pressure) -
           get(cold_eos_.pressure_from_density(rest_mass_density))) /
          get(rest_mass_density)};
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType>
HybridEos<ColdEquationOfState>::temperature_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy) const {
  return Scalar<DataType>{(thermal_adiabatic_index_ - 1.0) *
                          (get(specific_internal_energy) -
                           get(cold_eos_.specific_internal_energy_from_density(
                               rest_mass_density)))};
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType> HybridEos<ColdEquationOfState>::
    specific_internal_energy_from_density_and_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& temperature) const {
  return Scalar<DataType>{
      get(cold_eos_.specific_internal_energy_from_density(rest_mass_density)) +
      get(temperature) / (thermal_adiabatic_index_ - 1.0)};
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType>
HybridEos<ColdEquationOfState>::chi_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy) const {
  return Scalar<DataType>{
      get(cold_eos_.chi_from_density(rest_mass_density)) +
      (thermal_adiabatic_index_ - 1.0) *
          (get(specific_internal_energy) -
           get(cold_eos_.specific_internal_energy_from_density(
               rest_mass_density)) -
           get(cold_eos_.pressure_from_density(rest_mass_density)) /
               get(rest_mass_density))};
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType> HybridEos<ColdEquationOfState>::
    kappa_times_p_over_rho_squared_from_density_and_energy_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& specific_internal_energy) const {
  return Scalar<DataType>{
      (thermal_adiabatic_index_ - 1.0) *
          get(cold_eos_.pressure_from_density(rest_mass_density)) /
          get(rest_mass_density) +
      square(thermal_adiabatic_index_ - 1.0) *
          (get(specific_internal_energy) -
           get(cold_eos_.specific_internal_energy_from_density(
               rest_mass_density)))};
}
}  // namespace EquationsOfState

template class EquationsOfState::HybridEos<
    EquationsOfState::PolytropicFluid<true>>;
template class EquationsOfState::HybridEos<
    EquationsOfState::PolytropicFluid<false>>;
