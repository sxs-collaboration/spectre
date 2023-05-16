// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
}  // namespace domain
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace gh {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spatial derivatives of the shift vector from
 *        the generalized harmonic and geometric variables
 *
 * \details Spatial derivatives of the shift vector \f$\beta^i\f$ can be derived
 * from the following steps:
 * \f{align*}
 * \partial_i \beta^j
 *  =& \gamma^{jl} \gamma_{kl} \partial_i \beta^k \\
 *  =& \gamma^{jl} (\beta^k \partial_i \gamma_{lk}
 *         + \gamma_{kl}\partial_i \beta^k - \beta^k \partial_i \gamma_{kl}) \\
 *  =& \gamma^{jl} (\partial_i \beta_l - \beta^k \partial_i \gamma_{lk})
 *         (\because \gamma^{j0} = 0) \\
 *  =& \gamma^{ja} (\partial_i g_{a0} - \beta^k \partial _i g_{ak}) \\
 *  =& \alpha \gamma^{ja} n^b \partial_i g_{ab} \\
 *  =& (\gamma^{ja} - n^j n^a) \alpha n^b \Phi_{iab}
 *         - 2 n^j \partial_i \alpha \\
 *  =& g^{ja} \alpha n^b \Phi_{iab} - 2 n^j \partial_i \alpha \\
 *  =& \alpha (g^{ja} + n^j n^a) n^b \Phi_{iab}.
 * \f}
 * where we used the equation from spatial_deriv_of_lapse() for
 * \f$\partial_i \alpha\f$.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void spatial_deriv_of_shift(
    gsl::not_null<tnsr::iJ<DataType, SpatialDim, Frame>*> deriv_shift,
    const Scalar<DataType>& lapse,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::iJ<DataType, SpatialDim, Frame> spatial_deriv_of_shift(
    const Scalar<DataType>& lapse,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);
/// @}

namespace Tags {
/*!
 * \brief Compute item to get spatial derivatives of the shift vector from
 *        generalized harmonic and geometric variables
 *
 * \details See `spatial_deriv_of_shift()`. Can be retrieved using
 * `gr::Tags::Shift` wrapped in `::Tags::deriv`.
 */
template <size_t SpatialDim, typename Frame>
struct DerivShiftCompute
    : ::Tags::deriv<gr::Tags::Shift<DataVector, SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>,
      gr::Tags::InverseSpacetimeMetric<DataVector, SpatialDim, Frame>,
      gr::Tags::SpacetimeNormalVector<DataVector, SpatialDim, Frame>,
      Phi<DataVector, SpatialDim, Frame>>;

  using return_type = tnsr::iJ<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::iJ<DataVector, SpatialDim, Frame>*>,
      const Scalar<DataVector>&, const tnsr::AA<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&)>(
      &spatial_deriv_of_shift<DataVector, SpatialDim, Frame>);

  using base = ::Tags::deriv<gr::Tags::Shift<DataVector, SpatialDim, Frame>,
                             tmpl::size_t<SpatialDim>, Frame>;
};
}  // namespace Tags
}  // namespace gh
