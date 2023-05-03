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
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute inverse spacetime metric from inverse spatial metric, lapse
 * and shift
 *
 * \details The inverse spacetime metric \f$ g^{ab} \f$ is calculated as
 * \f{align}
 *    g^{tt} &= -  1/\alpha^2 \\
 *    g^{ti} &= \beta^i / \alpha^2 \\
 *    g^{ij} &= \gamma^{ij} - \beta^i \beta^j / \alpha^2
 * \f}
 * where \f$ \alpha, \beta^i\f$ and \f$ \gamma^{ij}\f$ are the lapse, shift and
 * inverse spatial metric respectively
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void inverse_spacetime_metric(
    gsl::not_null<tnsr::AA<DataType, SpatialDim, Frame>*>
        inverse_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::AA<DataType, SpatialDim, Frame> inverse_spacetime_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);
/// @}

namespace Tags {
/*!
 * \brief Compute item for inverse spacetime metric \f$g^{ab}\f$ in terms of the
 * lapse \f$\alpha\f$, shift \f$\beta^i\f$, and inverse spatial metric
 * \f$\gamma^{ij}\f$.
 *
 * \details Can be retrieved using `gr::Tags::InverseSpacetimeMetric`.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
struct InverseSpacetimeMetricCompute
    : InverseSpacetimeMetric<DataType, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<Lapse<DataType>, Shift<DataType, SpatialDim, Frame>,
                 InverseSpatialMetric<DataType, SpatialDim, Frame>>;

  using return_type = tnsr::AA<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::AA<DataType, SpatialDim, Frame>*>,
      const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
      const tnsr::II<DataType, SpatialDim, Frame>&)>(
      &inverse_spacetime_metric<DataType, SpatialDim, Frame>);

  using base = InverseSpacetimeMetric<DataType, SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace gr
