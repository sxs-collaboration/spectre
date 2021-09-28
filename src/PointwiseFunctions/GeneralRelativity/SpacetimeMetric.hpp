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
 * \details The spacetime metric \f$ \psi_{ab} \f$ is calculated as
 * \f{align}{
 *   \psi_{tt} &= - N^2 + N^m N^n g_{mn} \\
 *   \psi_{ti} &= g_{mi} N^m  \\
 *   \psi_{ij} &= g_{ij}
 * \f}
 * where \f$ N, N^i\f$ and \f$ g_{ij}\f$ are the lapse, shift and spatial metric
 * respectively
 */
template <size_t Dim, typename Frame, typename DataType>
void spacetime_metric(
    gsl::not_null<tnsr::aa<DataType, Dim, Frame>*> spacetime_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric);

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> spacetime_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric);
/// @}

namespace Tags {
/*!
 * \brief Compute item for spacetime metric \f$\psi_{ab}\f$ from the
 * lapse \f$N\f$, shift \f$N^i\f$, and spatial metric \f$g_{ij}\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpacetimeMetric`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpacetimeMetricCompute : SpacetimeMetric<SpatialDim, Frame, DataType>,
                                db::ComputeTag {
  using argument_tags =
      tmpl::list<Lapse<DataType>, Shift<SpatialDim, Frame, DataType>,
                 SpatialMetric<SpatialDim, Frame, DataType>>;

  using return_type = tnsr::aa<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*>,
      const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
      const tnsr::ii<DataType, SpatialDim, Frame>&)>(
      &spacetime_metric<SpatialDim, Frame, DataType>);

  using base = SpacetimeMetric<SpatialDim, Frame, DataType>;
};
}  // namespace Tags
}  // namespace gr
