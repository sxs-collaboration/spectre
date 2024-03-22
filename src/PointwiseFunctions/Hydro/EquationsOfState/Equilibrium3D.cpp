// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"

#include <memory>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/DarkEnergyFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Enthalpy.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace EquationsOfState {

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename EquilEos>,
                                     Equilibrium3D<EquilEos>, double, 3)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename EquilEos>,
                                     Equilibrium3D<EquilEos>, DataVector, 3)
template <typename EquilEos>
void Equilibrium3D<EquilEos>::pup(PUP::er& p) {
  EquationOfState<EquilEos::is_relativistic, 3>::pup(p);
  p | underlying_eos_;
}
template <typename EquilEos>
Equilibrium3D<EquilEos>::Equilibrium3D(CkMigrateMessage* msg)
    : EquationOfState<EquilEos::is_relativistic, 3>(msg) {}

template <typename EquilEos>
std::unique_ptr<EquationOfState<EquilEos::is_relativistic, 3>>
Equilibrium3D<EquilEos>::get_clone() const {
  auto clone = std::make_unique<Equilibrium3D<EquilEos>>(*this);
  return std::unique_ptr<EquationOfState<is_relativistic, 3>>(std::move(clone));
}

template <typename EquilEos>
bool Equilibrium3D<EquilEos>::is_equal(
    const EquationOfState<EquilEos::is_relativistic, 3>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const Equilibrium3D<EquilEos>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <typename EquilEos>
bool Equilibrium3D<EquilEos>::operator==(
    const Equilibrium3D<EquilEos>& rhs) const {
  return this->underlying_eos_ == rhs.underlying_eos_;
}
template <typename EquilEos>
bool Equilibrium3D<EquilEos>::operator!=(
    const Equilibrium3D<EquilEos>& rhs) const {
  return !(*this == rhs);
}

template <typename EquilEos>
template <class DataType>
Scalar<DataType>
Equilibrium3D<EquilEos>::pressure_from_density_and_temperature_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& temperature,
    const Scalar<DataType>& /*electron_fraction*/) const {
  return underlying_eos_.pressure_from_density_and_energy(
      rest_mass_density,
      underlying_eos_.specific_internal_energy_from_density_and_temperature(
          rest_mass_density, temperature));
}
template <typename EquilEos>
template <class DataType>
Scalar<DataType> Equilibrium3D<EquilEos>::pressure_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& /*electron_fraction*/) const {
  return underlying_eos_.pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy);
}
template <typename EquilEos>
template <class DataType>
Scalar<DataType>
Equilibrium3D<EquilEos>::temperature_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& /*electron_fraction*/) const {
  return underlying_eos_.temperature_from_density_and_energy(
      rest_mass_density, specific_internal_energy);
}

template <typename EquilEos>
template <class DataType>
Scalar<DataType> Equilibrium3D<EquilEos>::
    specific_internal_energy_from_density_and_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& temperature,
        const Scalar<DataType>& /*electron_fraction*/) const {
  return underlying_eos_.specific_internal_energy_from_density_and_temperature(
      rest_mass_density, temperature);
}

template <typename EquilEos>
template <class DataType>
Scalar<DataType>
Equilibrium3D<EquilEos>::sound_speed_squared_from_density_and_temperature_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& temperature,
    const Scalar<DataType>& /*electron_fraction*/) const {
  const Scalar<DataType> specific_internal_energy =
      underlying_eos_.specific_internal_energy_from_density_and_temperature(
          rest_mass_density, temperature);
  const DataType pressure =
      get(underlying_eos_.pressure_from_density_and_energy(
          rest_mass_density, specific_internal_energy));
  const DataType enthalpy_density =
      pressure + (1.0 + get(specific_internal_energy)) * get(rest_mass_density);
  // Cold part plus temperature dependent part, see the 3D EoS documentation for
  // expression.
  return Scalar<DataType>{
      get(rest_mass_density) / enthalpy_density *
      (get(underlying_eos_.chi_from_density_and_energy(
           rest_mass_density, specific_internal_energy)) +
       get(underlying_eos_
               .kappa_times_p_over_rho_squared_from_density_and_energy(
                   rest_mass_density, specific_internal_energy)))};
}

template class Equilibrium3D<HybridEos<PolytropicFluid<true>>>;
template class Equilibrium3D<HybridEos<PolytropicFluid<false>>>;
template class Equilibrium3D<HybridEos<Spectral>>;
template class Equilibrium3D<HybridEos<Enthalpy<Spectral>>>;
template class Equilibrium3D<DarkEnergyFluid<true>>;
template class Equilibrium3D<IdealFluid<true>>;
template class Equilibrium3D<IdealFluid<false>>;

}  // namespace EquationsOfState
