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
 * \brief Compute lapse from shift and spacetime metric
 *
 * \details Computes
 * \f{align}
 *    N &= \sqrt{N^i \psi_{it}-\psi_{tt}}
 * \f}
 * where \f$ N \f$, \f$ N^i\f$, and \f$\psi_{ab}\f$ are the lapse, shift,
 * and spacetime metric.
 * This can be derived, e.g., from Eqs. 2.121--2.122 of Baumgarte & Shapiro.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> lapse(
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void lapse(
    gsl::not_null<Scalar<DataType>*> lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item for lapse \f$N\f$ from the spacetime metric
 * \f$\psi_{ab}\f$ and the shift \f$N^i\f$.
 *
 * \details Can be retrieved using `gr::Tags::Lapse`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct LapseCompute : Lapse<DataType>, db::ComputeTag {
  using argument_tags =
      tmpl::list<Shift<SpatialDim, Frame, DataType>,
                 SpacetimeMetric<SpatialDim, Frame, DataType>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<Scalar<DataType>*> lapse,
      const tnsr::I<DataType, SpatialDim, Frame>&,
      const tnsr::aa<DataType, SpatialDim, Frame>&) noexcept>(
      &lapse<SpatialDim, Frame, DataType>);

  using base = Lapse<DataType>;
};
}  // namespace Tags
}  // namespace gr
