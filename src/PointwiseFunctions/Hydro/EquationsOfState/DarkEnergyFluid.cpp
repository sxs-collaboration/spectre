// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/DarkEnergyFluid.hpp"

#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

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
std::unique_ptr<EquationOfState<IsRelativistic, 3>>
DarkEnergyFluid<IsRelativistic>::promote_to_3d_eos() const {
  return std::make_unique<Equilibrium3D<DarkEnergyFluid<IsRelativistic>>>(
      *this);
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
Scalar<DataType>
DarkEnergyFluid<IsRelativistic>::specific_entropy_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy) const {
  if constexpr (std::is_same_v<DataType, double>) {
    return Scalar<double>{specific_entropy_from_density_and_energy(
        get(rest_mass_density), get(specific_internal_energy))};
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    auto result = make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
    for (size_t i = 0; i < get(result).size(); ++i) {
      get(result)[i] = specific_entropy_from_density_and_energy(
          get(rest_mass_density)[i], get(specific_internal_energy)[i]);
    }
    return result;
  }
}

template <bool IsRelativistic>
template <class DataType>
Scalar<DataType> DarkEnergyFluid<IsRelativistic>::
    specific_entropy_from_density_and_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& temperature) const {
  if constexpr (std::is_same_v<DataType, double>) {
    return Scalar<double>{specific_entropy_from_density_and_energy(
        get(rest_mass_density), get(temperature) / parameter_w_)};
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    auto result = make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
    for (size_t i = 0; i < get(result).size(); ++i) {
      get(result)[i] = specific_entropy_from_density_and_energy(
          get(rest_mass_density)[i], get(temperature)[i] / parameter_w_);
    }
    return result;
  }
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

template <bool IsRelativistic>
double
DarkEnergyFluid<IsRelativistic>::specific_entropy_from_density_and_energy(
    const double rest_mass_density,
    const double specific_internal_energy) const {
  // Note: Since specific internal energy has a lower bound of -1, entropy will
  // be undefined for values smaller than 0 or w = 0. Since this equation of
  // state is almost never used, this function is included as a framework for
  // the calculation. However, more careful attention is needed for users who
  // wish to use it.
  ASSERT(specific_internal_energy > 0.0,
         "Entropy is undefined for non-positive specific internal energy.");
  ASSERT(parameter_w_ > 0.0, "Entropy is undefined for $w = 0$.");
  return log(specific_internal_energy / pow(rest_mass_density, parameter_w_)) /
         parameter_w_;
}
}  // namespace EquationsOfState

template class EquationsOfState::DarkEnergyFluid<true>;
