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
 * \brief  Computes spacetime normal vector from lapse and shift.
 *
 * \details If \f$\alpha, \beta^i\f$ are the lapse and shift respectively, then
 *
 * \f{align} n^t &= 1/\alpha \\
 * n^i &= -\frac{\beta^i}{\alpha} \f}
 *
 * is computed.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::A<DataType, SpatialDim, Frame> spacetime_normal_vector(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift);

template <typename DataType, size_t SpatialDim, typename Frame>
void spacetime_normal_vector(
    gsl::not_null<tnsr::A<DataType, SpatialDim, Frame>*>
        spacetime_normal_vector,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift);
/// @}

namespace Tags {
/*!
 * \brief Compute item for spacetime normal vector \f$n^a\f$ from
 * the lapse \f$\alpha\f$ and the shift \f$\beta^i\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpacetimeNormalVector`.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
struct SpacetimeNormalVectorCompute
    : SpacetimeNormalVector<DataType, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<Lapse<DataType>, Shift<DataType, SpatialDim, Frame>>;

  using return_type = tnsr::A<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::A<DataType, SpatialDim, Frame>*>,
      const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&)>(
      &spacetime_normal_vector<DataType, SpatialDim, Frame>);

  using base = SpacetimeNormalVector<DataType, SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace gr
