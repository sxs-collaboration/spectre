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
 * \brief Computes time derivative of the shift vector from
 *        the generalized harmonic and geometric variables
 *
 * \details The time derivative of \f$ N^i \f$ can be derived from the following
 * steps:
 * \f{align*}
 * \partial_0 N^i
 *  =& g^{ik} \partial_0 (g_{kj} N^j) - N^j g^{ik} \partial_0 g_{kj} \\
 *  =& N g^{ik} t^b \partial_0 \psi_{kb} \\
 *  =& N g^{ik} t^b (\partial_0 - N^j\partial_j) \psi_{kb}
 *                  + N g^{ik} t^b N^j\partial_j \psi_{kb} \\
 *  =& -N^2 t^b\Pi_{kb} g^{ik} + N N^j t^b\Phi_{jkb} g^{ik} \\
 *  =& -N g^{ik} t^b (N \Pi_{kb} - N^j \Phi_{jkb}) \\
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_shift(
    gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*> dt_shift,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::I<DataType, SpatialDim, Frame> time_deriv_of_shift(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item to get time derivative of the shift vector from
 *        the generalized harmonic and geometric variables
 *
 * \details See `time_deriv_of_shift()`. Can be retrieved using
 * `gr::Tags::Shift` wrapped in `Tags::dt`.
 */
template <size_t SpatialDim, typename Frame>
struct TimeDerivShiftCompute
    : ::Tags::dt<gr::Tags::Shift<SpatialDim, Frame, DataVector>>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                 gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataVector>,
                 gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
                 Phi<SpatialDim, Frame>, Pi<SpatialDim, Frame>>;

  using return_type = tnsr::I<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::I<DataVector, SpatialDim, Frame>*>,
      const Scalar<DataVector>&, const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::II<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&) noexcept>(
      &time_deriv_of_shift<SpatialDim, Frame, DataVector>);

  using base = ::Tags::dt<gr::Tags::Shift<SpatialDim, Frame, DataVector>>;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
