// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
///// \endcond

namespace hydro {
/// @{
/*!
 * \brief Computes the inverse plasma beta
 *
 * The inverse plasma beta \f$\beta^{-1} = b^2 / (2 p)\f$, where
 * \f$b^2\f$ is the square of the comoving magnetic field amplitude
 * and \f$p\f$ is the fluid pressure.
 */
template <typename DataType>
void inverse_plasma_beta(
    gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& comoving_magnetic_field_magnitude,
    const Scalar<DataType>& fluid_pressure);

template <typename DataType>
Scalar<DataType> inverse_plasma_beta(
    const Scalar<DataType>& comoving_magnetic_field_magnitude,
    const Scalar<DataType>& fluid_pressure);
/// @}

namespace Tags {
/// Can be retrieved using `hydro::Tags::InversePlasmaBeta`
template <typename DataType>
struct InversePlasmaBetaCompute : InversePlasmaBeta<DataType>, db::ComputeTag {
  using base = InversePlasmaBeta<DataType>;
  using return_type = Scalar<DataType>;

  using argument_tags =
      tmpl::list<ComovingMagneticFieldMagnitude<DataType>, Pressure<DataType>>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const Scalar<DataType>&,
      const Scalar<DataType>&)>(&inverse_plasma_beta<DataType>);
};
}  // namespace Tags
}  // namespace hydro
