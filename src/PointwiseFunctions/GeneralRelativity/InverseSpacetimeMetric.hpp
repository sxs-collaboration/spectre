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
//@{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute inverse spacetime metric from inverse spatial metric, lapse
 * and shift
 *
 * \details The inverse spacetime metric \f$ \psi^{ab} \f$ is calculated as
 * \f{align}
 *    \psi^{tt} &= -  1/N^2 \\
 *    \psi^{ti} &= N^i / N^2 \\
 *    \psi^{ij} &= g^{ij} - N^i N^j / N^2
 * \f}
 * where \f$ N, N^i\f$ and \f$ g^{ij}\f$ are the lapse, shift and inverse
 * spatial metric respectively
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void inverse_spacetime_metric(
    gsl::not_null<tnsr::AA<DataType, SpatialDim, Frame>*>
        inverse_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::AA<DataType, SpatialDim, Frame> inverse_spacetime_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;
//@}

namespace Tags {
/*!
 * \brief Compute item for inverse spacetime metric \f$\psi^{ab}\f$
 * in terms of the lapse \f$N\f$, shift \f$N^i\f$, and inverse
 * spatial metric \f$g^{ij}\f$.
 *
 * \details Can be retrieved using `gr::Tags::InverseSpacetimeMetric`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct InverseSpacetimeMetricCompute
    : InverseSpacetimeMetric<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<Lapse<DataType>, Shift<SpatialDim, Frame, DataType>,
                 InverseSpatialMetric<SpatialDim, Frame, DataType>>;

  using return_type = tnsr::AA<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::AA<DataType, SpatialDim, Frame>*>,
      const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
      const tnsr::II<DataType, SpatialDim, Frame>&) noexcept>(
      &inverse_spacetime_metric<SpatialDim, Frame, DataType>);

  using base = InverseSpacetimeMetric<SpatialDim, Frame, DataType>;
};
}  // namespace Tags
}  // namespace gr
