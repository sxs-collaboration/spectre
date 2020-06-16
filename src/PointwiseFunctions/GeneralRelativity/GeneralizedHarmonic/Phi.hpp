// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
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
// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the auxiliary variable \f$\Phi_{iab}\f$ used by the
 * generalized harmonic formulation of Einstein's equations.
 *
 * \details If \f$ N, N^i\f$ and \f$ g_{ij} \f$ are the lapse, shift and spatial
 * metric respectively, then \f$\Phi_{iab} \f$ is computed as
 *
 * \f{align}
 *     \Phi_{ktt} &= - 2 N \partial_k N
 *                 + 2 g_{mn} N^m \partial_k N^n
 *                 + N^m N^n \partial_k g_{mn} \\
 *     \Phi_{kti} &= g_{mi} \partial_k N^m
 *                 + N^m \partial_k g_{mi} \\
 *     \Phi_{kij} &= \partial_k g_{ij}
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void phi(gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> phi,
         const Scalar<DataType>& lapse,
         const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
         const tnsr::I<DataType, SpatialDim, Frame>& shift,
         const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
         const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
         const tnsr::ijj<DataType, SpatialDim, Frame>&
             deriv_spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> phi(
    const Scalar<DataType>& lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept;
// @}

namespace Tags {
/*!
 * \brief Compute item for the auxiliary variable \f$\Phi_{iab}\f$ used by the
 * generalized harmonic formulation of Einstein's equations.
 *
 * \details See `phi()`. Can be retrieved using
 * `GeneralizedHarmonic::Tags::Phi`.
 */
template <size_t SpatialDim, typename Frame>
struct PhiCompute : Phi<SpatialDim, Frame>, db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<SpatialDim>,
                    Frame>,
      gr::Tags::Shift<SpatialDim, Frame, DataVector>,
      ::Tags::deriv<gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>,
      gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
      ::Tags::deriv<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>>;

  using return_type = tnsr::iaa<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::iaa<DataVector, SpatialDim, Frame>*>,
      const Scalar<DataVector>&, const tnsr::i<DataVector, SpatialDim, Frame>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::iJ<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::ijj<DataVector, SpatialDim, Frame>&) noexcept>(
      &phi<SpatialDim, Frame, DataVector>);

  using base = Phi<SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
