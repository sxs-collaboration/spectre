// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic2D.hpp"

#include <memory>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Enthalpy.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PiecewisePolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace EquationsOfState {

EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEos>,
                                     Barotropic2D<ColdEos>, double, 2)
EQUATION_OF_STATE_MEMBER_DEFINITIONS(template <typename ColdEos>,
                                     Barotropic2D<ColdEos>, DataVector, 2)

template <typename ColdEos>
std::unique_ptr<EquationOfState<ColdEos::is_relativistic, 2>>
Barotropic2D<ColdEos>::get_clone() const {
  auto clone = std::make_unique<Barotropic2D<ColdEos>>(*this);
  return std::unique_ptr<EquationOfState<is_relativistic, 2>>(std::move(clone));
}

template <typename ColdEos>
std::unique_ptr<EquationOfState<ColdEos::is_relativistic, 3>>
Barotropic2D<ColdEos>::promote_to_3d_eos() const {
  return std::make_unique<Barotropic3D<ColdEos>>(underlying_eos_);
}

template <typename ColdEos>
bool Barotropic2D<ColdEos>::is_equal(
    const EquationOfState<ColdEos::is_relativistic, 2>& rhs) const {
  const auto& derived_ptr =
      dynamic_cast<const Barotropic2D<ColdEos>* const>(&rhs);
  return derived_ptr != nullptr and *derived_ptr == *this;
}

template <typename ColdEos>
void Barotropic2D<ColdEos>::pup(PUP::er& p) {
  EquationOfState<ColdEos::is_relativistic, 2>::pup(p);
  p | underlying_eos_;
}
template <typename ColdEos>
Barotropic2D<ColdEos>::Barotropic2D(CkMigrateMessage* msg)
    : EquationOfState<ColdEos::is_relativistic, 2>(msg) {}

template <typename ColdEos>
bool Barotropic2D<ColdEos>::operator==(const Barotropic2D<ColdEos>& rhs) const {
  return this->underlying_eos_ == rhs.underlying_eos_;
}
template <typename ColdEos>
bool Barotropic2D<ColdEos>::operator!=(const Barotropic2D<ColdEos>& rhs) const {
  return not(*this == rhs);
}

template <typename ColdEos>
template <class DataType>
Scalar<DataType> Barotropic2D<ColdEos>::pressure_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& /*specific_internal_energy*/) const {
  return underlying_eos_.pressure_from_density(rest_mass_density);
}

template <typename ColdEos>
template <class DataType>
Scalar<DataType> Barotropic2D<ColdEos>::pressure_from_density_and_enthalpy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& /*specific_enthalpy*/) const {
  return underlying_eos_.pressure_from_density(rest_mass_density);
}

template <typename ColdEos>
template <class DataType>
Scalar<DataType>
Barotropic2D<ColdEos>::specific_internal_energy_from_density_and_pressure_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& /*pressure*/) const {
  return underlying_eos_.specific_internal_energy_from_density(
      rest_mass_density);
}

template <typename ColdEos>
template <class DataType>
Scalar<DataType>
Barotropic2D<ColdEos>::temperature_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& /*specific_internal_energy*/) const {
  return make_with_value<Scalar<DataType>>(rest_mass_density, 0.0);
}

template <typename ColdEos>
template <class DataType>
Scalar<DataType> Barotropic2D<ColdEos>::
    specific_internal_energy_from_density_and_temperature_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& /*temperature*/) const {
  return underlying_eos_.specific_internal_energy_from_density(
      rest_mass_density);
}

template <typename ColdEos>
template <class DataType>
Scalar<DataType> Barotropic2D<ColdEos>::chi_from_density_and_energy_impl(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& /*specific_internal_energy*/) const {
  return underlying_eos_.chi_from_density(rest_mass_density);
}

template <typename ColdEos>
template <class DataType>
Scalar<DataType> Barotropic2D<ColdEos>::
    kappa_times_p_over_rho_squared_from_density_and_energy_impl(
        const Scalar<DataType>& rest_mass_density,
        const Scalar<DataType>& /*specific_internal_energy*/) const {
  return underlying_eos_.kappa_times_p_over_rho_squared_from_density(
      rest_mass_density);
}

template class Barotropic2D<EquationsOfState::PolytropicFluid<true>>;
template class Barotropic2D<EquationsOfState::PolytropicFluid<false>>;
template class Barotropic2D<PiecewisePolytropicFluid<true>>;
template class Barotropic2D<PiecewisePolytropicFluid<false>>;
template class Barotropic2D<Spectral>;
template class Barotropic2D<Enthalpy<PolytropicFluid<true>>>;
template class Barotropic2D<Enthalpy<Spectral>>;
template class Barotropic2D<Enthalpy<Enthalpy<Spectral>>>;
template class Barotropic2D<Enthalpy<Enthalpy<Enthalpy<Spectral>>>>;

}  // namespace EquationsOfState
