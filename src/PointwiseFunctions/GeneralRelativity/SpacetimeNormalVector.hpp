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
 * \details If \f$N, N^i\f$ are the lapse and shift respectively, then
 * \f{align} n^t &= 1/N \\
 * n^i &= -\frac{N^i}{N} \f}
 * is computed.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::A<DataType, SpatialDim, Frame> spacetime_normal_vector(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_normal_vector(
    gsl::not_null<tnsr::A<DataType, SpatialDim, Frame>*>
        spacetime_normal_vector,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item for spacetime normal vector \f$n^a\f$ from
 * the lapse \f$N\f$ and the shift \f$N^i\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpacetimeNormalVector`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpacetimeNormalVectorCompute
    : SpacetimeNormalVector<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<Lapse<DataType>, Shift<SpatialDim, Frame, DataType>>;

  using return_type = tnsr::A<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::A<DataType, SpatialDim, Frame>*>,
      const Scalar<DataType>&,
      const tnsr::I<DataType, SpatialDim, Frame>&) noexcept>(
      &spacetime_normal_vector<SpatialDim, Frame, DataType>);

  using base = SpacetimeNormalVector<SpatialDim, Frame, DataType>;
};
}  // namespace Tags
}  // namespace gr
