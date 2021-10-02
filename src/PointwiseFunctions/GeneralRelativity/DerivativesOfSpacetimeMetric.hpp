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
 *     \partial_\mu \psi_{tt} &= - 2 N \partial_\mu N
 *                 + 2 g_{mn} N^m \partial_\mu N^n
 *                 + N^m N^n \partial_\mu g_{mn} \\
 *     \partial_\mu \psi_{ti} &= g_{mi} \partial_\mu N^m
 *                 + N^m \partial_\mu g_{mi} \\
 *     \partial_\mu \psi_{ij} &= \partial_\mu g_{ij}
 * \f}
 * where \f$ N, N^i, g \f$ are the lapse, shift, and spatial metric
 * respectively.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
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

template <size_t SpatialDim, typename Frame, typename DataType>
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
    : gr::Tags::DerivativesOfSpacetimeMetric<SpatialDim, Frame, DataVector>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<SpatialDim>,
                    Frame>,
      gr::Tags::Shift<SpatialDim, Frame, DataVector>,
      ::Tags::dt<gr::Tags::Shift<SpatialDim, Frame, DataVector>>,
      ::Tags::deriv<gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                    tmpl::size_t<SpatialDim>, Frame>,
      gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
      ::Tags::dt<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>>,
      ::Tags::deriv<gr::Tags::SpatialMetric<SpatialDim, Frame, DataVector>,
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
      &gr::derivatives_of_spacetime_metric<SpatialDim, Frame, DataVector>);

  using base =
      gr::Tags::DerivativesOfSpacetimeMetric<SpatialDim, Frame, DataVector>;
};
}  // namespace Tags
}  // namespace gr
