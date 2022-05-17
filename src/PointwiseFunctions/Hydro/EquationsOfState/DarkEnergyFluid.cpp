// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/DarkEnergyFluid.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace EquationsOfState {
template <bool IsRelativistic>
DarkEnergyFluid<IsRelativistic>::DarkEnergyFluid(const double parameter_w)
    : parameter_w_(parameter_w) {
  if (parameter_w_ <= 0.0 or parameter_w_ > 1.0) {
    ERROR("The w(z) parameter must be positive, but less than one");
  }
}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     DarkEnergyFluid<IsRelativistic>, double, 2)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <bool IsRelativistic>,
                                     DarkEnergyFluid<IsRelativistic>,
                                     DataVector, 2)

template <bool IsRelativistic>
std::unique_ptr<EquationOfState<IsRelativistic, 2>>
DarkEnergyFluid<IsRelativistic>::get_clone() const {
  auto clone = std::make_unique<DarkEnergyFluid<IsRelativistic>>(*this);
  return std::unique_ptr<EquationOfState<IsRelativistic, 2>>(std::move(clone));
}

template <bool IsRelativistic>
bool DarkEnergyFluid<IsRelativistic>::is_equal(
    const EquationOfState<IsRelativistic, 2>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const DarkEnergyFluid<IsRelativistic>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <bool IsRelativistic>
bool DarkEnergyFluid<IsRelativistic>::operator==(
    const DarkEnergyFluid<IsRelativistic>& rhs) const {
  return parameter_w_ == rhs.parameter_w_;
}

template <bool IsRelativistic>
bool DarkEnergyFluid<IsRelativistic>::operator!=(
    const DarkEnergyFluid<IsRelativistic>& rhs) const {
  return not(*this == rhs);
}

template <bool IsRelativistic>
DarkEnergyFluid<IsRelativistic>::DarkEnergyFluid(CkMigrateMessage* msg)
    : EquationOfState<IsRelativistic, 2>(msg) {}

template <bool IsRelativistic>
void DarkEnergyFluid<IsRelativistic>::pup(PUP::er& p) {
  EquationOfState<IsRelativistic, 2>::pup(p);
  p | parameter_w_;
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
DarkEnergyFluid<IsRelativistic>::pressure_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy) const {
  return Scalar<DataType>{parameter_w_ * get(rest_mass_density) *
                          (1.0 + get(specific_internal_energy))};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
DarkEnergyFluid<IsRelativistic>::pressure_from_density_and_enthalpy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy) const {
  return Scalar<DataType>{(parameter_w_ / (parameter_w_ + 1.0)) *
                          get(rest_mass_density) * get(specific_enthalpy)};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> DarkEnergyFluid<IsRelativistic>::
    specific_internal_energy_from_density_and_pressure_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& pressure) const {
  return Scalar<DataType>{
      get(pressure) / (parameter_w_ * get(rest_mass_density)) - 1.0};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
DarkEnergyFluid<IsRelativistic>::temperature_from_density_and_energy_impl(
    const Scalar<DataType>& /*rest_mass_density*/,
    const Scalar<DataType>& specific_internal_energy) const {
  return Scalar<DataType>{parameter_w_ * get(specific_internal_energy)};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> DarkEnergyFluid<IsRelativistic>::
    specific_internal_energy_from_density_and_temperature_impl(
        const Scalar<DataType>& /*rest_mass_density*/,
        const Scalar<DataType>& temperature) const {
  return Scalar<DataType>{get(temperature) / parameter_w_};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType>
DarkEnergyFluid<IsRelativistic>::chi_from_density_and_energy_impl(
    const Scalar<DataType>& /*rest_mass_density*/,
    const Scalar<DataType>& specific_internal_energy) const {
  return Scalar<DataType>{parameter_w_ * (1.0 + get(specific_internal_energy))};
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> DarkEnergyFluid<IsRelativistic>::
    kappa_times_p_over_rho_squared_from_density_and_energy_impl(
        const Scalar<DataType>& /*rest_mass_density*/,
        const Scalar<DataType>& specific_internal_energy) const {
  return Scalar<DataType>{square(parameter_w_) *
                          (1.0 + get(specific_internal_energy))};
}
}  // namespace EquationsOfState

template class EquationsOfState::DarkEnergyFluid<true>;
