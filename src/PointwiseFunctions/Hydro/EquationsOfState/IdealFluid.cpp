// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace EquationsOfState {
template <bool IsRelativistic>
IdealFluid<IsRelativistic>::IdealFluid(const double adiabatic_index) noexcept
    : adiabatic_index_(adiabatic_index) {}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     IdealFluid<IsRelativistic>, double, 2)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     IdealFluid<IsRelativistic>, DataVector, 2)

template <bool IsRelativistic>
IdealFluid<IsRelativistic>::IdealFluid(CkMigrateMessage* /*unused*/) noexcept {}

template <bool IsRelativistic>
void IdealFluid<IsRelativistic>::pup(PUP::er& p) noexcept {
  EquationOfState<IsRelativistic, 2>::pup(p);
  p | adiabatic_index_;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
IdealFluid<IsRelativistic>::pressure_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy) const noexcept {
  return Scalar<DataType>{get(rest_mass_density) *
                          get(specific_internal_energy) *
                          (adiabatic_index_ - 1.0)};
}

template <>
template <class DataType>
Scalar<DataType> IdealFluid<true>::pressure_from_density_and_enthalpy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy) const noexcept {
  return Scalar<DataType>{get(rest_mass_density) *
                          (get(specific_enthalpy) - 1.0) *
                          (adiabatic_index_ - 1.0) / adiabatic_index_};
}

template <>
template <class DataType>
Scalar<DataType> IdealFluid<false>::pressure_from_density_and_enthalpy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy) const noexcept {
  return Scalar<DataType>{get(rest_mass_density) * get(specific_enthalpy) *
                          (adiabatic_index_ - 1.0) / adiabatic_index_};
}

template <>
template <class DataType>
Scalar<DataType>
IdealFluid<true>::specific_enthalpy_from_density_and_energy_impl(
    const Scalar<DataType>& /*rest_mass_density*/,
    const Scalar<DataType>& specific_internal_energy) const noexcept {
  return Scalar<DataType>{1.0 +
                          adiabatic_index_ * get(specific_internal_energy)};
}

template <>
template <class DataType>
Scalar<DataType>
IdealFluid<false>::specific_enthalpy_from_density_and_energy_impl(
    const Scalar<DataType>& /*rest_mass_density*/,
    const Scalar<DataType>& specific_internal_energy) const noexcept {
  return Scalar<DataType>{adiabatic_index_ * get(specific_internal_energy)};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> IdealFluid<IsRelativistic>::
    specific_internal_energy_from_density_and_pressure_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& pressure) const noexcept {
  return Scalar<DataType>{1.0 / (adiabatic_index_ - 1.0) * get(pressure) /
                          get(rest_mass_density)};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> IdealFluid<IsRelativistic>::chi_from_density_and_energy_impl(
    const Scalar<DataType>& /*rest_mass_density*/,
    const Scalar<DataType>& specific_internal_energy) const noexcept {
  return Scalar<DataType>{get(specific_internal_energy) *
                          (adiabatic_index_ - 1.0)};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> IdealFluid<IsRelativistic>::
    kappa_times_p_over_rho_squared_from_density_and_energy_impl(
        const Scalar<DataType>& /*rest_mass_density*/,
        const Scalar<DataType>& specific_internal_energy) const noexcept {
  return Scalar<DataType>{square(adiabatic_index_ - 1.0) *
                          get(specific_internal_energy)};
}
}  // namespace EquationsOfState

template class EquationsOfState::IdealFluid<true>;
template class EquationsOfState::IdealFluid<false>;
/// \endcond
