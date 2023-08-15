// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"

#include <memory>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Enthalpy.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace EquationsOfState {

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEquilEos>,
                                     Barotropic3D<ColdEquilEos>, double, 3)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEquilEos>,
                                     Barotropic3D<ColdEquilEos>, DataVector, 3)

template <typename ColdEquilEos>
std::unique_ptr<EquationOfState<ColdEquilEos::is_relativistic, 3>>
Barotropic3D<ColdEquilEos>::get_clone() const {
  auto clone = std::make_unique<Barotropic3D<ColdEquilEos>>(*this);
  return std::unique_ptr<EquationOfState<is_relativistic, 3>>(std::move(clone));
}

template <typename ColdEquilEos>
bool Barotropic3D<ColdEquilEos>::is_equal(
    const EquationOfState<ColdEquilEos::is_relativistic, 3>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const Barotropic3D<ColdEquilEos>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <typename ColdEquilEos>
void Barotropic3D<ColdEquilEos>::pup(PUP::er& p) {
  EquationOfState<ColdEquilEos::is_relativistic, 3>::pup(p);
  p | underlying_eos_;
}
template <typename ColdEquilEos>
Barotropic3D<ColdEquilEos>::Barotropic3D(CkMigrateMessage* msg)
    : EquationOfState<ColdEquilEos::is_relativistic, 3>(msg) {}

template <typename ColdEquilEos>
bool Barotropic3D<ColdEquilEos>::operator==(
    const Barotropic3D<ColdEquilEos>& rhs) const {
  return this->underlying_eos_ == rhs.underlying_eos_;
}
template <typename ColdEquilEos>
bool Barotropic3D<ColdEquilEos>::operator!=(
    const Barotropic3D<ColdEquilEos>& rhs) const {
  return not(*this == rhs);
}

template <typename ColdEquilEos>
template <class DataType>
Scalar<DataType>
Barotropic3D<ColdEquilEos>::pressure_from_density_and_temperature_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& /*temperature*/,
    const Scalar<DataType>& /*electron_fraction*/) const {
  return underlying_eos_.pressure_from_density(rest_mass_density);
}
template <typename ColdEquilEos>
template <class DataType>
Scalar<DataType>
Barotropic3D<ColdEquilEos>::pressure_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& /*specific_internal_energy*/,
    const Scalar<DataType>& /*electron_fraction*/) const {
  return underlying_eos_.pressure_from_density(rest_mass_density);
}
template <typename ColdEquilEos>
template <class DataType>
Scalar<DataType>
Barotropic3D<ColdEquilEos>::temperature_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& /*specific_internal_energy*/,
    const Scalar<DataType>& /*electron_fraction*/) const {
  return make_with_value<Scalar<DataType>>(rest_mass_density, 0.0);
}

template <typename ColdEquilEos>
template <class DataType>
Scalar<DataType> Barotropic3D<ColdEquilEos>::
    specific_internal_energy_from_density_and_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& /*temperature*/,
        const Scalar<DataType>& /*electron_fraction*/) const {
  return underlying_eos_.specific_internal_energy_from_density(
      rest_mass_density);
}
template <typename ColdEquilEos>
template <class DataType>
Scalar<DataType> Barotropic3D<ColdEquilEos>::
    sound_speed_squared_from_density_and_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& /*temperature*/,
        const Scalar<DataType>& /*electron_fraction*/) const {
  // We have to do this to avoid dividing by zero
  const DataType enthalpy_density =
      get(underlying_eos_.pressure_from_density(rest_mass_density)) +
      (1.0 + get(underlying_eos_.specific_internal_energy_from_density(
                 rest_mass_density))) *
          get(rest_mass_density);
  return Scalar<DataType>{
      get(rest_mass_density) *
      get(underlying_eos_.chi_from_density(rest_mass_density)) /
      enthalpy_density};
}
template class Barotropic3D<EquationsOfState::PolytropicFluid<true>>;
template class Barotropic3D<Spectral>;
template class Barotropic3D<Enthalpy<Spectral>>;

}  // namespace EquationsOfState
