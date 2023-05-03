// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spacetime derivative of spacetime metric from spatial metric,
 * lapse, shift, and their space and time derivatives.
 *
 * \details Computes the derivatives as:
 * \f{align}
 *     \partial_\mu g_{tt} &= - 2 \alpha \partial_\mu \alpha
 *                 + 2 \gamma_{mn} \beta^m \partial_\mu \beta^n
 *                 + \beta^m \beta^n \partial_\mu \gamma_{mn} \\
 *     \partial_\mu g_{ti} &= \gamma_{mi} \partial_\mu \beta^m
 *                 + \beta^m \partial_\mu \gamma_{mi} \\
 *     \partial_\mu g_{ij} &= \partial_\mu \gamma_{ij}
 * \f}
 * where \f$ \alpha, \beta^i, \gamma_{ij} \f$ are the lapse, shift, and spatial
 * metric respectively.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void derivatives_of_spacetime_metric(
    gsl::not_null<tnsr::abb<DataType, SpatialDim, Frame>*>
        spacetime_deriv_spacetime_metric,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& deriv_spatial_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::abb<DataType, SpatialDim, Frame> derivatives_of_spacetime_metric(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& deriv_spatial_metric);
/// @}

namespace Tags {
/*!
 * \brief Compute item to get spacetime derivative of spacetime metric from
 * spatial metric, lapse, shift, and their space and time derivatives.
 *
 * \details See `derivatives_of_spacetime_metric()`. Can be retrieved using
 * `gr::Tags::DerivativesOfSpacetimeMetric`.
 */
template <size_t SpatialDim, typename Frame>
struct DerivativesOfSpacetimeMetricCompute
    : gr::Tags::DerivativesOfSpacetimeMetric<DataVector, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<SpatialDim>,
                    Frame>,
      gr::Tags::Shift<DataVector, SpatialDim, Frame>,
      ::Tags::dt<gr::Tags::Shift<DataVector, SpatialDim, Frame>>,
      ::Tags::deriv<gr::Tags::Shift<DataVector, SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>,
      gr::Tags::SpatialMetric<DataVector, SpatialDim, Frame>,
      ::Tags::dt<gr::Tags::SpatialMetric<DataVector, SpatialDim, Frame>>,
      ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>>;

  using return_type = tnsr::abb<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::abb<DataVector, SpatialDim, Frame>*>
          spacetime_deriv_spacetime_metric,
      const Scalar<DataVector>&, const Scalar<DataVector>&,
      const tnsr::i<DataVector, SpatialDim, Frame>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::iJ<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::ijj<DataVector, SpatialDim, Frame>&)>(
      &gr::derivatives_of_spacetime_metric<DataVector, SpatialDim, Frame>);

  using base =
      gr::Tags::DerivativesOfSpacetimeMetric<DataVector, SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace gr
