// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
namespace gr {
// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute spatial metric from spacetime metric.
 * \details Simply pull out the spatial components.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> spatial_metric(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_metric(
    gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept;
// @}

namespace Tags {
/*!
 * \brief Compute item for spatial metric \f$g_{ij}\f$ from the
 * spacetime metric \f$\psi_{ab}\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpatialMetric`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpatialMetricCompute : SpatialMetric<SpatialDim, Frame, DataType>,
                              db::ComputeTag {
  using argument_tags =
      tmpl::list<SpacetimeMetric<SpatialDim, Frame, DataType>>;

  using return_type = tnsr::ii<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>,
      const tnsr::aa<DataType, SpatialDim, Frame>&) noexcept>(
      &spatial_metric<SpatialDim, Frame, DataType>);

  using base = SpatialMetric<SpatialDim, Frame, DataType>;
};
}  // namespace Tags
}  // namespace gr
