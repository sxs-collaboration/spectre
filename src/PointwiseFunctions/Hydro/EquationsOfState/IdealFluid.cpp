// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"

#include <limits>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace EquationsOfState {
template <bool IsRelativistic>
IdealFluid<IsRelativistic>::IdealFluid(const double adiabatic_index)
    : adiabatic_index_(adiabatic_index) {}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     IdealFluid<IsRelativistic>, double, 2)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     IdealFluid<IsRelativistic>, DataVector, 2)
template <bool IsRelativistic>
std::unique_ptr<EquationOfState<IsRelativistic, 2>>
IdealFluid<IsRelativistic>::get_clone() const {
  auto clone = std::make_unique<IdealFluid<IsRelativistic>>(*this);
  return std::unique_ptr<EquationOfState<IsRelativistic, 2>>(std::move(clone));
}

template <bool IsRelativistic>
bool IdealFluid<IsRelativistic>::is_equal(
    const EquationOfState<IsRelativistic, 2>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const IdealFluid<IsRelativistic>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <bool IsRelativistic>
bool IdealFluid<IsRelativistic>::operator==(
    const IdealFluid<IsRelativistic>& rhs) const {
  return adiabatic_index_ == rhs.adiabatic_index_;
}

template <bool IsRelativistic>
bool IdealFluid<IsRelativistic>::operator!=(
    const IdealFluid<IsRelativistic>& rhs) const {
  return not(*this == rhs);
}

template <bool IsRelativistic>
IdealFluid<IsRelativistic>::IdealFluid(CkMigrateMessage* msg)
    : EquationOfState<IsRelativistic, 2>(msg) {}

template <bool IsRelativistic>
void IdealFluid<IsRelativistic>::pup(PUP::er& p) {
  EquationOfState<IsRelativistic, 2>::pup(p);
  p | adiabatic_index_;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
IdealFluid<IsRelativistic>::pressure_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy) const {
  return Scalar<DataType>{get(rest_mass_density) *
                          get(specific_internal_energy) *
                          (adiabatic_index_ - 1.0)};
}

template <>
template <class DataType>
Scalar<DataType> IdealFluid<true>::pressure_from_density_and_enthalpy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy) const {
  return Scalar<DataType>{get(rest_mass_density) *
                          (get(specific_enthalpy) - 1.0) *
                          (adiabatic_index_ - 1.0) / adiabatic_index_};
}

template <>
template <class DataType>
Scalar<DataType> IdealFluid<false>::pressure_from_density_and_enthalpy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy) const {
  return Scalar<DataType>{get(rest_mass_density) * get(specific_enthalpy) *
                          (adiabatic_index_ - 1.0) / adiabatic_index_};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> IdealFluid<IsRelativistic>::
    specific_internal_energy_from_density_and_pressure_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& pressure) const {
  return Scalar<DataType>{1.0 / (adiabatic_index_ - 1.0) * get(pressure) /
                          get(rest_mass_density)};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
IdealFluid<IsRelativistic>::temperature_from_density_and_energy_impl(
    const Scalar<DataType>& /*rest_mass_density*/,
    const Scalar<DataType>& specific_internal_energy) const {
  return Scalar<DataType>{(adiabatic_index_ - 1.0) *
                          get(specific_internal_energy)};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> IdealFluid<IsRelativistic>::
    specific_internal_energy_from_density_and_temperature_impl(
        const Scalar<DataType>& /*rest_mass_density*/,
        const Scalar<DataType>& temperature) const {
  return Scalar<DataType>{get(temperature) / (adiabatic_index_ - 1.0)};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> IdealFluid<IsRelativistic>::chi_from_density_and_energy_impl(
    const Scalar<DataType>& /*rest_mass_density*/,
    const Scalar<DataType>& specific_internal_energy) const {
  return Scalar<DataType>{get(specific_internal_energy) *
                          (adiabatic_index_ - 1.0)};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> IdealFluid<IsRelativistic>::
    kappa_times_p_over_rho_squared_from_density_and_energy_impl(
        const Scalar<DataType>& /*rest_mass_density*/,
        const Scalar<DataType>& specific_internal_energy) const {
  return Scalar<DataType>{square(adiabatic_index_ - 1.0) *
                          get(specific_internal_energy)};
}

template <bool IsRelativistic>
double IdealFluid<IsRelativistic>::specific_internal_energy_upper_bound(
    const double /* rest_mass_density */) const {
  // this bound comes from the dominant energy condition which implies
  // that the pressure is bounded by the total energy density,
  // i.e. p < e = rho * (1 + eps)
  if (IsRelativistic and adiabatic_index_ > 2.0) {
    return 1.0 / (adiabatic_index_ - 2.0);
  }
  return std::numeric_limits<double>::max();
}
}  // namespace EquationsOfState

template class EquationsOfState::IdealFluid<true>;
template class EquationsOfState::IdealFluid<false>;
