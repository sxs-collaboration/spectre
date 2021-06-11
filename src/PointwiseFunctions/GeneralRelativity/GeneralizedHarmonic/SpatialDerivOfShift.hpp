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

namespace GeneralizedHarmonic {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spatial derivatives of the shift vector from
 *        the generalized harmonic and geometric variables
 *
 * \details Spatial derivatives of the shift vector \f$N^i\f$ can be derived
 * from the following steps:
 * \f{align*}
 * \partial_i N^j
 *  =& g^{jl} g_{kl} \partial_i N^k \\
 *  =& g^{jl} (N^k \partial_i g_{lk}
 *             + g_{kl}\partial_i N^k - N^k \partial_i g_{kl}) \\
 *  =& g^{jl} (\partial_i N_l - N^k \partial_i g_{lk}) (\because g^{j0} = 0) \\
 *  =& g^{ja} (\partial_i \psi_{a0} - N^k \partial _i \psi_{ak}) \\
 *  =& N g^{ja} t^b \partial_i \psi_{ab} \\
 *  =& (g^{ja} - t^j t^a) N t^b \Phi_{iab} - 2 t^j \partial_i N \\
 *  =& \psi^{ja} N t^b \Phi_{iab} - 2 t^j \partial_i N \\
 *  =& N (\psi^{ja} + t^j t^a) t^b \Phi_{iab}.
 * \f}
 * where we used the equation from spatial_deriv_of_lapse() for
 * \f$\partial_i N\f$.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_deriv_of_shift(
    gsl::not_null<tnsr::iJ<DataType, SpatialDim, Frame>*> deriv_shift,
    const Scalar<DataType>& lapse,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iJ<DataType, SpatialDim, Frame> spatial_deriv_of_shift(
    const Scalar<DataType>& lapse,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;
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
    : ::Tags::deriv<gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>,
      gr::Tags::InverseSpacetimeMetric<SpatialDim, Frame, DataVector>,
      gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
      Phi<SpatialDim, Frame>>;

  using return_type = tnsr::iJ<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::iJ<DataVector, SpatialDim, Frame>*>,
      const Scalar<DataVector>&, const tnsr::AA<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&) noexcept>(
      &spatial_deriv_of_shift<SpatialDim, Frame, DataVector>);

  using base = ::Tags::deriv<gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                             tmpl::size_t<SpatialDim>, Frame>;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
