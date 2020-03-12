// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl

namespace hydro {
//@{
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Computes the relativistic specific enthalpy \f$h\f$ as:
 * \f$ h = 1 + \epsilon + \frac{p}{\rho} \f$
 * where \f$\epsilon\f$ is the specific internal energy, \f$p\f$
 * is the pressure, and \f$\rho\f$ is the rest mass density.
 */
template <typename DataType>
void relativistic_specific_enthalpy(
    gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure) noexcept;

template <typename DataType>
Scalar<DataType> relativistic_specific_enthalpy(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure) noexcept;
//@}

namespace Tags {
/// Compute item for the relativistic specific enthalpy \f$h\f$.
///
/// Can be retrieved using `hydro::Tags::SpecificEnthalpy`
template <typename DataType>
struct SpecificEnthalpyCompute : SpecificEnthalpy<DataType>, db::ComputeTag {
  using argument_tags =
      tmpl::list<RestMassDensity<DataType>, SpecificInternalEnergy<DataType>,
                 Pressure<DataType>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const Scalar<DataType>&,
      const Scalar<DataType>&, const Scalar<DataType>&) noexcept>(
      &relativistic_specific_enthalpy<DataType>);

  using base = SpecificEnthalpy<DataType>;
};
}  // namespace Tags
}  // namespace hydro
