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
#include "DataStructures/VariablesTag.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the spacetime metric from the spatial metric, lapse, and
 * shift.
 * \details The spacetime metric \f$ g_{ab} \f$ is calculated as
 * \f{align}{
 *   g_{tt} &= - \alpha^2 + \beta^m \beta^n \gamma_{mn} \\
 *   g_{ti} &= \gamma_{mi} \beta^m  \\
 *   g_{ij} &= \gamma_{ij}
 * \f}
 * where \f$ \alpha, \beta^i\f$ and \f$ \gamma_{ij}\f$ are the lapse, shift and
 * spatial metric respectively
 */
template <typename DataType, size_t Dim, typename Frame>
void spacetime_metric(
    gsl::not_null<tnsr::aa<DataType, Dim, Frame>*> spacetime_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::aa<DataType, SpatialDim, Frame> spacetime_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric);
/// @}

namespace Tags {
/*!
 * \brief Compute item for spacetime metric \f$g_{ab}\f$ from the lapse
 * \f$\alpha\f$, shift \f$\beta^i\f$, and spatial metric \f$\gamma_{ij}\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpacetimeMetric`.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
struct SpacetimeMetricCompute : SpacetimeMetric<DataType, SpatialDim, Frame>,
                                db::ComputeTag {
  using argument_tags =
      tmpl::list<Lapse<DataType>, Shift<DataType, SpatialDim, Frame>,
                 SpatialMetric<DataType, SpatialDim, Frame>>;

  using return_type = tnsr::aa<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*>,
      const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
      const tnsr::ii<DataType, SpatialDim, Frame>&)>(
      &spacetime_metric<DataType, SpatialDim, Frame>);

  using base = SpacetimeMetric<DataType, SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace gr
