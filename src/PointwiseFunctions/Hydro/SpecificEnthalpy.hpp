// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace hydro {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Computes the relativistic specific enthalpy \f$h\f$ as:
 * \f$ h = 1 + \epsilon + \frac{p}{\rho} \f$
 * where \f$\epsilon\f$ is the specific internal energy, \f$p\f$
 * is the pressure, and \f$\rho\f$ is the rest mass density.
 */
template <typename DataType>
Scalar<DataType> specific_enthalpy(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure) noexcept;

namespace Tags {
/// Compute item for specific enthalpy \f$h\f$.
///
/// Can be retrieved using `hydro::Tags::SpecificEnthalpy`
template <typename DataType>
struct SpecificEnthalpyCompute : SpecificEnthalpy<DataType>, db::ComputeTag {
  static constexpr auto function = &specific_enthalpy<DataType>;
  using argument_tags =
      tmpl::list<RestMassDensity<DataType>, SpecificInternalEnergy<DataType>,
                 Pressure<DataType>>;
};
}  // namespace Tags
}  // namespace hydro
