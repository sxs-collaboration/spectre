// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute spatial metric from spacetime metric.
 * \details Simply pull out the spatial components.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ii<DataType, SpatialDim, Frame> spatial_metric(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
void spatial_metric(
    gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute the induced spatial metric \f$\gamma_{ab}\f$ in spacetime
 * coordinates.
 * \details The induced spatial metric is \f$\gamma_{ab}=g_{ab} + n_a n_b\f$.
 * Since \f$n_a=(-\alpha,0,0,0)\f$, this adds \f$\alpha^2\f$ to \f$g_{tt}\f$.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::aa<DataType, SpatialDim, Frame> induced_spatial_metric(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const Scalar<DataType>& lapse);

template <typename DataType, size_t SpatialDim, typename Frame>
void induced_spatial_metric(
    gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*> result,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const Scalar<DataType>& lapse);
/// @}

namespace Tags {
/*!
 * \brief Compute item for spatial metric \f$\gamma_{ij}\f$ from the
 * spacetime metric \f$g_{ab}\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpatialMetric`.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
struct SpatialMetricCompute : SpatialMetric<DataType, SpatialDim, Frame>,
                              db::ComputeTag {
  using argument_tags =
      tmpl::list<SpacetimeMetric<DataType, SpatialDim, Frame>>;

  using return_type = tnsr::ii<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>,
      const tnsr::aa<DataType, SpatialDim, Frame>&)>(
      &spatial_metric<DataType, SpatialDim, Frame>);

  using base = SpatialMetric<DataType, SpatialDim, Frame>;
};
/*!
 * \brief Compute item for the induced spatial metric \f$\gamma_{ab}\f$
 * from the spacetime metric \f$g_{ab}\f$ and the spacetime normal one-form
 * \f$n_a\f$.
 *
 * \details Can be retrieved using `gr::Tags::InducedSpatialMetric`.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
struct InducedSpatialMetricCompute
    : InducedSpatialMetric<DataType, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<SpacetimeMetric<DataType, SpatialDim, Frame>, Lapse<DataType>>;

  using return_type = tnsr::aa<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*>,
      const tnsr::aa<DataType, SpatialDim, Frame>&, const Scalar<DataType>&)>(
      &induced_spatial_metric<DataType, SpatialDim, Frame>);

  using base = InducedSpatialMetric<DataType, SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace gr
