// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"

#include <memory>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Enthalpy.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace EquationsOfState {
template <typename ColdEquationOfState>
HybridEos<ColdEquationOfState>::HybridEos(ColdEquationOfState cold_eos,
                                          const double thermal_adiabatic_index,
                                          const double min_temperature)
    : cold_eos_(std::move(cold_eos)),
      thermal_adiabatic_index_(thermal_adiabatic_index),
      min_temperature_(min_temperature) {}

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEquationOfState>,
                                     HybridEos<ColdEquationOfState>, double, 2)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEquationOfState>,
                                     HybridEos<ColdEquationOfState>, DataVector,
                                     2)

template <typename ColdEquationOfState>
std::unique_ptr<
    EquationOfState<HybridEos<ColdEquationOfState>::is_relativistic, 2>>
HybridEos<ColdEquationOfState>::get_clone() const {
  auto clone = std::make_unique<HybridEos<ColdEquationOfState>>(*this);
  return std::unique_ptr<EquationOfState<is_relativistic, 2>>(std::move(clone));
}

template <typename ColdEquationOfState>
std::unique_ptr<
    EquationOfState<HybridEos<ColdEquationOfState>::is_relativistic, 3>>
HybridEos<ColdEquationOfState>::promote_to_3d_eos() const {
  return std::make_unique<Equilibrium3D<HybridEos<ColdEquationOfState>>>(
      Equilibrium3D(*this));
}

template <typename ColdEquationOfState>
bool HybridEos<ColdEquationOfState>::operator==(
    const HybridEos<ColdEquationOfState>& rhs) const {
  return cold_eos_ == rhs.cold_eos_ and
         thermal_adiabatic_index_ == rhs.thermal_adiabatic_index_ and
         min_temperature_ == rhs.min_temperature_;
}

template <typename ColdEquationOfState>
bool HybridEos<ColdEquationOfState>::operator!=(
    const HybridEos<ColdEquationOfState>& rhs) const {
  return not(*this == rhs);
}

template <typename ColdEquationOfState>
bool HybridEos<ColdEquationOfState>::is_equal(
    const EquationOfState<is_relativistic, 2>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const HybridEos<ColdEquationOfState>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <typename ColdEquationOfState>
HybridEos<ColdEquationOfState>::HybridEos(CkMigrateMessage* msg)
    : EquationOfState<is_relativistic, 2>(msg) {}

template <typename ColdEquationOfState>
void HybridEos<ColdEquationOfState>::pup(PUP::er& p) {
  EquationOfState<is_relativistic, 2>::pup(p);
  p | cold_eos_;
  p | thermal_adiabatic_index_;
  p | min_temperature_;
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType>
HybridEos<ColdEquationOfState>::pressure_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy) const {
  using std::max;
  return Scalar<DataType>{
      get(cold_eos_.pressure_from_density(rest_mass_density)) +
      get(rest_mass_density) * (thermal_adiabatic_index_ - 1.0) *
          max((get(specific_internal_energy) -
               get(cold_eos_.specific_internal_energy_from_density(
                   rest_mass_density))),
              0.0)};
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType>
HybridEos<ColdEquationOfState>::pressure_from_density_and_enthalpy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy) const {
  using std::max;
  if constexpr (ColdEquationOfState::is_relativistic) {
    return Scalar<DataType>{
        (get(cold_eos_.pressure_from_density(rest_mass_density)) +
         get(rest_mass_density) * (thermal_adiabatic_index_ - 1.0) *
             max((get(specific_enthalpy) - 1.0 -
                  get(cold_eos_.specific_internal_energy_from_density(
                      rest_mass_density))),
                 0.0)) /
        thermal_adiabatic_index_};
  } else {
    return Scalar<DataType>{
        (get(cold_eos_.pressure_from_density(rest_mass_density)) +
         get(rest_mass_density) * (thermal_adiabatic_index_ - 1.0) *
             max((get(specific_enthalpy) -
                  get(cold_eos_.specific_internal_energy_from_density(
                      rest_mass_density))),
                 0.0)) /
        thermal_adiabatic_index_};
  }
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType> HybridEos<ColdEquationOfState>::
    specific_internal_energy_from_density_and_pressure_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& pressure) const {
  using std::max;
  return Scalar<DataType>{
      get(cold_eos_.specific_internal_energy_from_density(rest_mass_density)) +
      1.0 / (thermal_adiabatic_index_ - 1.0) *
          max((get(pressure) -
               get(cold_eos_.pressure_from_density(rest_mass_density))),
              0.0) /
          get(rest_mass_density)};
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType>
HybridEos<ColdEquationOfState>::temperature_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy) const {
  using std::max;
  return Scalar<DataType>{
      (thermal_adiabatic_index_ - 1.0) *
      max((get(specific_internal_energy) -
           get(cold_eos_.specific_internal_energy_from_density(
               rest_mass_density))),
          0.0)};
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType>
HybridEos<ColdEquationOfState>::specific_entropy_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy) const {
  using std::max;
  DataType thermal_specific_internal_energy =
      max(get(specific_internal_energy) -
              get(cold_eos_.specific_internal_energy_from_density(
                  rest_mass_density)),
          0.0);
  if constexpr (std::is_same_v<DataType, double>) {
    return Scalar<double>{specific_entropy_from_density_and_thermal_energy(
        get(rest_mass_density), thermal_specific_internal_energy)};
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    auto result = make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
    for (size_t i = 0; i < get(result).size(); ++i) {
      get(result)[i] = specific_entropy_from_density_and_thermal_energy(
          get(rest_mass_density)[i], thermal_specific_internal_energy[i]);
    }
    return result;
  }
}

template <typename ColdEquationOfState>
template <class DataType>
Scalar<DataType> HybridEos<ColdEquationOfState>::
    specific_entropy_from_density_and_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& temperature) const {
  if constexpr (std::is_same_v<DataType, double>) {
    return Scalar<double>{specific_entropy_from_density_and_thermal_energy(
        get(rest_mass_density),
        get(temperature) / (thermal_adiabatic_index_ - 1.0))};
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    auto result = make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
    for (size_t i = 0; i < get(result).size(); ++i) {
      get(result)[i] = specific_entropy_from_density_and_thermal_energy(
          get(rest_mass_density)[i],
          get(temperature)[i] / (thermal_adiabatic_index_ - 1.0));
    }
    return result;
  }
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
  using std::max;
  return Scalar<DataType>{
      (thermal_adiabatic_index_ - 1.0) *
          get(cold_eos_.pressure_from_density(rest_mass_density)) /
          get(rest_mass_density) +
      square(thermal_adiabatic_index_ - 1.0) *
          max((get(specific_internal_energy) -
               get(cold_eos_.specific_internal_energy_from_density(
                   rest_mass_density))),
              0.0)};
}

template <typename ColdEquationOfState>
double HybridEos<ColdEquationOfState>::
    specific_entropy_from_density_and_thermal_energy(
        const double rest_mass_density,
        const double thermal_specific_internal_energy) const {
  using std::max;
  const double floored_thermal_specific_internal_energy =
      max(min_temperature_ / (thermal_adiabatic_index_ - 1.0),
          thermal_specific_internal_energy);
  ASSERT(floored_thermal_specific_internal_energy > 0.0,
         "The entropy is only well defined for positive nonzero temperatures. "
         "Try setting the minimum temperature to be greater than 0.");
  return log(floored_thermal_specific_internal_energy /
             pow(rest_mass_density, thermal_adiabatic_index_ - 1.0)) /
         (thermal_adiabatic_index_ - 1.0);
}
}  // namespace EquationsOfState

template class EquationsOfState::HybridEos<
    EquationsOfState::PolytropicFluid<true>>;
template class EquationsOfState::HybridEos<
    EquationsOfState::PolytropicFluid<false>>;
template class EquationsOfState::HybridEos<EquationsOfState::Spectral>;
template class EquationsOfState::HybridEos<
    EquationsOfState::Enthalpy<EquationsOfState::PolytropicFluid<true>>>;
template class EquationsOfState::HybridEos<
    EquationsOfState::Enthalpy<EquationsOfState::Enthalpy<
        EquationsOfState::Enthalpy<EquationsOfState::PolytropicFluid<true>>>>>;
template class EquationsOfState::HybridEos<
    EquationsOfState::Enthalpy<EquationsOfState::Spectral>>;
template class EquationsOfState::HybridEos<EquationsOfState::Enthalpy<
    EquationsOfState::Enthalpy<EquationsOfState::Spectral>>>;
template class EquationsOfState::HybridEos<
    EquationsOfState::Enthalpy<EquationsOfState::Enthalpy<
        EquationsOfState::Enthalpy<EquationsOfState::Spectral>>>>;
