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
 * \brief Compute shift from spacetime metric and inverse spatial metric.
 *
 * \details Computes
 * \f{align}
 *    N^i &= g^{ij} \psi_{jt}
 * \f}
 * where \f$ N^i\f$, \f$ g^{ij}\f$, and \f$\psi_{ab}\f$ are the shift, inverse
 * spatial metric, and spacetime metric.
 * This can be derived, e.g., from Eqs. 2.121--2.122 of Baumgarte & Shapiro.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::I<DataType, SpatialDim, Frame> shift(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void shift(gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*> shift,
           const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
           const tnsr::II<DataType, SpatialDim, Frame>&
               inverse_spatial_metric) noexcept;
// @}

namespace Tags {
/*!
 * \brief Compute item for shift \f$N^i\f$ from the spacetime metric
 * \f$\psi_{ab}\f$ and the inverse spatial metric \f$g^{ij}\f$.
 *
 * \details Can be retrieved using `gr::Tags::Shift`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct ShiftCompute : Shift<SpatialDim, Frame, DataType>, db::ComputeTag {
  using argument_tags =
      tmpl::list<SpacetimeMetric<SpatialDim, Frame, DataType>,
                 InverseSpatialMetric<SpatialDim, Frame, DataType>>;

  using return_type = tnsr::I<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*> shift,
      const tnsr::aa<DataType, SpatialDim, Frame>&,
      const tnsr::II<DataType, SpatialDim, Frame>&) noexcept>(
      &shift<SpatialDim, Frame, DataType>);

  using base = Shift<SpatialDim, Frame, DataType>;
};
}  // namespace Tags
}  // namespace gr
