// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/DarkEnergyFluid.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace EquationsOfState {
template <bool IsRelativistic>
DarkEnergyFluid<IsRelativistic>::DarkEnergyFluid(
    const double parameter_w) noexcept
    : parameter_w_(parameter_w) {
  if (parameter_w_ <= 0.0) {
    ERROR("The w(z) parameter must be positive.");
  }
}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     DarkEnergyFluid<IsRelativistic>, double, 2)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     DarkEnergyFluid<IsRelativistic>,
                                     DataVector, 2)

template <bool IsRelativistic>
DarkEnergyFluid<IsRelativistic>::DarkEnergyFluid(
    CkMigrateMessage* /*unused*/) noexcept {}

template <bool IsRelativistic>
void DarkEnergyFluid<IsRelativistic>::pup(PUP::er& p) noexcept {
  EquationOfState<IsRelativistic, 2>::pup(p);
  p | parameter_w_;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
DarkEnergyFluid<IsRelativistic>::pressure_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy) const noexcept {
  return Scalar<DataType>{parameter_w_ * get(rest_mass_density) *
                          (1.0 + get(specific_internal_energy))};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
DarkEnergyFluid<IsRelativistic>::pressure_from_density_and_enthalpy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy) const noexcept {
  return Scalar<DataType>{(parameter_w_ / (parameter_w_ + 1.0)) *
                          get(rest_mass_density) * get(specific_enthalpy)};
}

template <>
template <class DataType>
Scalar<DataType>
DarkEnergyFluid<true>::specific_enthalpy_from_density_and_energy_impl(
    const Scalar<DataType>& /*rest_mass_density*/,
    const Scalar<DataType>& specific_internal_energy) const noexcept {
  return Scalar<DataType>{(1.0 + parameter_w_) *
                          (1.0 + get(specific_internal_energy))};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> DarkEnergyFluid<IsRelativistic>::
    specific_internal_energy_from_density_and_pressure_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& pressure) const noexcept {
  return Scalar<DataType>{
      get(pressure) / (parameter_w_ * get(rest_mass_density)) - 1.0};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
DarkEnergyFluid<IsRelativistic>::chi_from_density_and_energy_impl(
    const Scalar<DataType>& /*rest_mass_density*/,
    const Scalar<DataType>& specific_internal_energy) const noexcept {
  return Scalar<DataType>{parameter_w_ * (1.0 + get(specific_internal_energy))};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> DarkEnergyFluid<IsRelativistic>::
    kappa_times_p_over_rho_squared_from_density_and_energy_impl(
        const Scalar<DataType>& /*rest_mass_density*/,
        const Scalar<DataType>& specific_internal_energy) const noexcept {
  return Scalar<DataType>{square(parameter_w_) *
                          (1.0 + get(specific_internal_energy))};
}
}  // namespace EquationsOfState

template class EquationsOfState::DarkEnergyFluid<true>;
/// \endcond
